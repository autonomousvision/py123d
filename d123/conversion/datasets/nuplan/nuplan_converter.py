import os
import pickle
import uuid
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, Union

import numpy as np
import yaml

import d123.conversion.datasets.nuplan.utils as nuplan_utils
from d123.common.utils.dependencies import check_dependencies
from d123.conversion.abstract_dataset_converter import AbstractDatasetConverter
from d123.conversion.dataset_converter_config import DatasetConverterConfig
from d123.conversion.datasets.nuplan.nuplan_map_conversion import NuPlanMapConverter
from d123.conversion.datasets.nuplan.utils.nuplan_constants import (
    NUPLAN_DATA_SPLITS,
    NUPLAN_DEFAULT_DT,
    NUPLAN_MAP_LOCATIONS,
    NUPLAN_ROLLING_SHUTTER_S,
    NUPLAN_TRAFFIC_STATUS_DICT,
)
from d123.conversion.datasets.nuplan.utils.nuplan_sql_helper import (
    get_box_detections_for_lidarpc_token_from_db,
    get_nearest_ego_pose_for_timestamp_from_db,
)
from d123.conversion.log_writer.abstract_log_writer import AbstractLogWriter
from d123.conversion.utils.sensor_utils.lidar_index_registry import NuPlanLidarIndex
from d123.datatypes.detections.detection import (
    BoxDetectionSE3,
    BoxDetectionWrapper,
    TrafficLightDetection,
    TrafficLightDetectionWrapper,
)
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from d123.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3
from d123.datatypes.vehicle_state.vehicle_parameters import (
    get_nuplan_chrysler_pacifica_parameters,
    rear_axle_se3_to_center_se3,
)
from d123.geometry import StateSE3, Vector3D

check_dependencies(["nuplan", "sqlalchemy"], "nuplan")
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras, get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.simulation.observation.observation_type import CameraChannel

# NOTE: Leaving this constant here, to avoid having a nuplan dependency in nuplan_constants.py
NUPLAN_CAMERA_MAPPING = {
    PinholeCameraType.CAM_F0: CameraChannel.CAM_F0,
    PinholeCameraType.CAM_B0: CameraChannel.CAM_B0,
    PinholeCameraType.CAM_L0: CameraChannel.CAM_L0,
    PinholeCameraType.CAM_L1: CameraChannel.CAM_L1,
    PinholeCameraType.CAM_L2: CameraChannel.CAM_L2,
    PinholeCameraType.CAM_R0: CameraChannel.CAM_R0,
    PinholeCameraType.CAM_R1: CameraChannel.CAM_R1,
    PinholeCameraType.CAM_R2: CameraChannel.CAM_R2,
}

TARGET_DT: Final[float] = 0.1  # TODO: make configurable


def create_splits_logs() -> Dict[str, List[str]]:
    # NOTE: nuPlan stores the training and validataion logs
    yaml_filepath = Path(nuplan_utils.__path__[0]) / "log_splits.yaml"
    with open(yaml_filepath, "r") as stream:
        splits = yaml.safe_load(stream)

    return splits["log_splits"]


class NuPlanConverter(AbstractDatasetConverter):
    def __init__(
        self,
        splits: List[str],
        nuplan_data_root: Union[Path, str],
        nuplan_map_root: Union[Path, str],
        nuplan_sensor_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        super().__init__(dataset_converter_config)

        for split in splits:
            assert (
                split in NUPLAN_DATA_SPLITS
            ), f"Split {split} is not available. Available splits: {NUPLAN_DATA_SPLITS}"

        self._splits: List[str] = splits
        self._nuplan_data_root: Path = Path(nuplan_data_root)
        self._nuplan_map_root: Path = Path(nuplan_map_root)
        self._nuplan_sensor_root: Path = Path(nuplan_sensor_root)

        self._split_log_path_pairs: List[Tuple[str, List[Path]]] = self._collect_split_log_path_pairs()

    def _collect_split_log_path_pairs(self) -> List[Tuple[str, List[Path]]]:
        # NOTE: the nuplan mini folder has an internal train, val, test structure, all stored in "mini".
        # The complete dataset is saved in the "trainval" folder (train and val), or in the "test" folder (for test).
        # Thus, we need filter the logs in a split, based on the internal nuPlan configuration.
        split_log_path_pairs: List[Tuple[str, List[Path]]] = []
        log_names_per_split = create_splits_logs()

        for split in self._splits:
            split_type = split.split("_")[-1]
            assert split_type in ["train", "val", "test"]

            if split in ["nuplan_train", "nuplan_val"]:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "trainval"
            elif split in ["nuplan_test"]:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "test"
            elif split in ["nuplan_mini_train", "nuplan_mini_val", "nuplan_mini_test"]:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "mini"
            elif split == "nuplan_private_test":
                # TODO: Remove private split
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "private_test"

            all_log_files_in_path = [log_file for log_file in nuplan_split_folder.glob("*.db")]

            if split == "nuplan_private_test":
                # TODO: Remove private split
                valid_log_names = [str(log_file.stem) for log_file in all_log_files_in_path]
            else:
                all_log_files_in_path = [log_file for log_file in nuplan_split_folder.glob("*.db")]
                all_log_names = set([str(log_file.stem) for log_file in all_log_files_in_path])
                log_names_in_split = set(log_names_per_split[split_type])
                valid_log_names = list(all_log_names & log_names_in_split)

            for log_name in valid_log_names:
                log_path = nuplan_split_folder / f"{log_name}.db"
                split_log_path_pairs.append((split, log_path))

        return split_log_path_pairs

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(NUPLAN_MAP_LOCATIONS)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._split_log_path_pairs)

    def convert_map(self, map_index: int) -> None:
        """Inherited, see superclass."""
        map_name = NUPLAN_MAP_LOCATIONS[map_index]
        map_path = self.dataset_converter_config.output_path / "maps" / f"nuplan_{map_name}.gpkg"
        if self.dataset_converter_config.force_map_conversion or not map_path.exists():
            map_path.unlink(missing_ok=True)
            NuPlanMapConverter(
                nuplan_map_root=self._nuplan_map_root,
                map_path=self.dataset_converter_config.output_path / "maps",
            ).convert(map_name=map_name)

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""
        int(os.environ.get("NODE_RANK", 0))
        str(uuid.uuid4())

        split, source_log_path = self._split_log_path_pairs[log_index]

        nuplan_log_db = NuPlanDB(self._nuplan_data_root, str(source_log_path), None)

        log_name = nuplan_log_db.log_name

        # 1. Initialize log metadata
        log_metadata = LogMetadata(
            dataset="nuplan",
            split=split,
            log_name=log_name,
            location=nuplan_log_db.log.map_version,
            timestep_seconds=TARGET_DT,
            vehicle_parameters=get_nuplan_chrysler_pacifica_parameters(),
            camera_metadata=_get_nuplan_camera_metadata(source_log_path, self.dataset_converter_config),
            lidar_metadata=_get_nuplan_lidar_metadata(
                self._nuplan_sensor_root, log_name, self.dataset_converter_config
            ),
            map_has_z=False,
            map_is_local=False,
        )

        # 2. Prepare log writer
        overwrite_log = log_writer.reset(self.dataset_converter_config, log_metadata)

        if overwrite_log:
            step_interval: float = int(TARGET_DT / NUPLAN_DEFAULT_DT)
            for nuplan_lidar_pc in nuplan_log_db.lidar_pc[::step_interval]:

                lidar_pc_token: str = nuplan_lidar_pc.token
                log_writer.write(
                    timestamp=TimePoint.from_us(nuplan_lidar_pc.timestamp),
                    ego_state=_extract_nuplan_ego_state(nuplan_lidar_pc),
                    box_detections=_extract_nuplan_box_detections(nuplan_lidar_pc, source_log_path),
                    traffic_lights=_extract_nuplan_traffic_lights(nuplan_log_db, lidar_pc_token),
                    cameras=_extract_nuplan_cameras(
                        nuplan_log_db=nuplan_log_db,
                        nuplan_lidar_pc=nuplan_lidar_pc,
                        source_log_path=source_log_path,
                        nuplan_sensor_root=self._nuplan_sensor_root,
                        dataset_converter_config=self.dataset_converter_config,
                    ),
                    lidars=_extract_nuplan_lidars(
                        nuplan_lidar_pc=nuplan_lidar_pc,
                        nuplan_sensor_root=self._nuplan_sensor_root,
                        dataset_converter_config=self.dataset_converter_config,
                    ),
                    scenario_tags=_extract_nuplan_scenario_tag(nuplan_log_db, lidar_pc_token),
                    route_lane_group_ids=_extract_nuplan_route_lane_group_ids(nuplan_lidar_pc),
                )
                del nuplan_lidar_pc

        log_writer.close()

        # NOTE: The nuPlanDB class has several internal references, which makes memory management tricky.
        # We need to ensure all references are released properly. It is not always working with just del.
        nuplan_log_db.detach_tables()
        nuplan_log_db.remove_ref()
        del nuplan_log_db


def _get_nuplan_camera_metadata(
    source_log_path: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, PinholeCameraMetadata]:

    def _get_camera_metadata(camera_type: PinholeCameraType) -> PinholeCameraMetadata:
        cam = list(get_cameras(source_log_path, [str(NUPLAN_CAMERA_MAPPING[camera_type].value)]))[0]

        intrinsics_camera_matrix = np.array(pickle.loads(cam.intrinsic), dtype=np.float64)  # array of shape (3, 3)
        intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsics_camera_matrix)

        distortion_array = np.array(pickle.loads(cam.distortion), dtype=np.float64)  # array of shape (5,)
        distortion = PinholeDistortion.from_array(distortion_array, copy=False)

        return PinholeCameraMetadata(
            camera_type=camera_type,
            width=cam.width,
            height=cam.height,
            intrinsics=intrinsic,
            distortion=distortion,
        )

    camera_metadata: Dict[str, PinholeCameraMetadata] = {}
    if dataset_converter_config.include_cameras:
        for camera_type in NUPLAN_CAMERA_MAPPING.keys():
            camera_metadata[camera_type] = _get_camera_metadata(camera_type)

    return camera_metadata


def _get_nuplan_lidar_metadata(
    nuplan_sensor_root: Path,
    log_name: str,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LiDARType, LiDARMetadata]:

    metadata: Dict[LiDARType, LiDARMetadata] = {}
    log_lidar_folder = nuplan_sensor_root / log_name / "MergedPointCloud"

    # NOTE: We first need to check if the LiDAR folder exists, as not all logs have LiDAR data
    if log_lidar_folder.exists() and log_lidar_folder.is_dir() and dataset_converter_config.include_lidars:
        metadata[LiDARType.LIDAR_MERGED] = LiDARMetadata(
            lidar_type=LiDARType.LIDAR_MERGED,
            lidar_index=NuPlanLidarIndex,
            extrinsic=None,  # NOTE: LiDAR extrinsic are unknown
        )
    return metadata


def _extract_nuplan_ego_state(nuplan_lidar_pc: LidarPc) -> EgoStateSE3:

    vehicle_parameters = get_nuplan_chrysler_pacifica_parameters()
    rear_axle_pose = StateSE3(
        x=nuplan_lidar_pc.ego_pose.x,
        y=nuplan_lidar_pc.ego_pose.y,
        z=nuplan_lidar_pc.ego_pose.z,
        qw=nuplan_lidar_pc.ego_pose.qw,
        qx=nuplan_lidar_pc.ego_pose.qx,
        qy=nuplan_lidar_pc.ego_pose.qy,
        qz=nuplan_lidar_pc.ego_pose.qz,
    )
    center = rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_pose, vehicle_parameters=vehicle_parameters)
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(
            x=nuplan_lidar_pc.ego_pose.vx,
            y=nuplan_lidar_pc.ego_pose.vy,
            z=nuplan_lidar_pc.ego_pose.vz,
        ),
        acceleration=Vector3D(
            x=nuplan_lidar_pc.ego_pose.acceleration_x,
            y=nuplan_lidar_pc.ego_pose.acceleration_y,
            z=nuplan_lidar_pc.ego_pose.acceleration_z,
        ),
        angular_velocity=Vector3D(
            x=nuplan_lidar_pc.ego_pose.angular_rate_x,
            y=nuplan_lidar_pc.ego_pose.angular_rate_y,
            z=nuplan_lidar_pc.ego_pose.angular_rate_z,
        ),
    )
    return EgoStateSE3(
        center_se3=center,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,  # NOTE: Timepoint is not needed during writing, set to None
    )


def _extract_nuplan_box_detections(lidar_pc: LidarPc, source_log_path: Path) -> BoxDetectionWrapper:
    box_detections: List[BoxDetectionSE3] = list(
        get_box_detections_for_lidarpc_token_from_db(source_log_path, lidar_pc.token)
    )
    return BoxDetectionWrapper(box_detections=box_detections)


def _extract_nuplan_traffic_lights(log_db: NuPlanDB, lidar_pc_token: str) -> TrafficLightDetectionWrapper:
    traffic_lights_detections: List[TrafficLightDetection] = [
        TrafficLightDetection(
            timepoint=None,  # NOTE: Timepoint is not needed during writing, set to None
            lane_id=int(traffic_light.lane_connector_id),
            status=NUPLAN_TRAFFIC_STATUS_DICT[traffic_light.status],
        )
        for traffic_light in log_db.traffic_light_status.select_many(lidar_pc_token=lidar_pc_token)
    ]
    return TrafficLightDetectionWrapper(traffic_light_detections=traffic_lights_detections)


def _extract_nuplan_cameras(
    nuplan_log_db: NuPlanDB,
    nuplan_lidar_pc: LidarPc,
    source_log_path: Path,
    nuplan_sensor_root: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]]:

    camera_dict: Dict[str, Union[str, bytes]] = {}

    if dataset_converter_config.include_cameras:
        log_cam_infos = {camera.token: camera for camera in nuplan_log_db.log.cameras}
        for camera_type, camera_channel in NUPLAN_CAMERA_MAPPING.items():
            camera_data: Optional[Union[str, bytes]] = None
            image_class = list(
                get_images_from_lidar_tokens(source_log_path, [nuplan_lidar_pc.token], [str(camera_channel.value)])
            )

            if len(image_class) != 0:
                image = image_class[0]
                filename_jpg = nuplan_sensor_root / image.filename_jpg
                if filename_jpg.exists() and filename_jpg.is_file():

                    # NOTE: This part of the modified from the MTGS code
                    # In MTGS, a slower method is used to find the nearest ego pose.
                    # The code below uses a direct SQL query to find the nearest ego pose, in a given window.
                    # https://github.com/OpenDriveLab/MTGS/blob/main/nuplan_scripts/utils/nuplan_utils_custom.py#L117

                    # Query nearest ego pose for the image timestamp
                    timestamp = image.timestamp + NUPLAN_ROLLING_SHUTTER_S.time_us
                    nearest_ego_pose = get_nearest_ego_pose_for_timestamp_from_db(
                        source_log_path,
                        timestamp,
                        [nuplan_lidar_pc.token],
                    )

                    # Compute camera to ego transformation, given the nearest ego pose
                    img_e2g = nearest_ego_pose.transformation_matrix
                    g2e = nuplan_lidar_pc.ego_pose.trans_matrix_inv
                    img_e2e = g2e @ img_e2g
                    cam_info = log_cam_infos[image.camera_token]
                    c2img_e = cam_info.trans_matrix
                    c2e = img_e2e @ c2img_e
                    extrinsic = StateSE3.from_transformation_matrix(c2e)

                    # Store camera data, either as path or binary
                    camera_data: Optional[Union[str, bytes]] = None
                    if dataset_converter_config.camera_store_option == "path":
                        camera_data = str(filename_jpg)
                    elif dataset_converter_config.camera_store_option == "binary":
                        with open(filename_jpg, "rb") as f:
                            camera_data = f.read()

                    # Store in dictionary
                    camera_dict[camera_type] = camera_data, extrinsic

    return camera_dict


def _extract_nuplan_lidars(
    nuplan_lidar_pc: LidarPc,
    nuplan_sensor_root: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LiDARType, Optional[str]]:

    lidar: Optional[str] = None
    lidar_full_path = nuplan_sensor_root / nuplan_lidar_pc.filename
    if lidar_full_path.exists() and lidar_full_path.is_file():
        lidar = nuplan_lidar_pc.filename

    return {LiDARType.LIDAR_MERGED: lidar}


def _extract_nuplan_scenario_tag(nuplan_log_db: NuPlanDB, lidar_pc_token: str) -> List[str]:
    scenario_tags = [
        scenario_tag.type for scenario_tag in nuplan_log_db.scenario_tag.select_many(lidar_pc_token=lidar_pc_token)
    ]
    if len(scenario_tags) == 0:
        scenario_tags = ["unknown"]
    return scenario_tags


def _extract_nuplan_route_lane_group_ids(nuplan_lidar_pc: LidarPc) -> List[int]:
    return [
        int(roadblock_id)
        for roadblock_id in str(nuplan_lidar_pc.scene.roadblock_ids).split(" ")
        if len(roadblock_id) > 0
    ]
