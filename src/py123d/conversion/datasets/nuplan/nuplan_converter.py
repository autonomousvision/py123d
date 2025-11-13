import pickle
from pathlib import Path
from typing import Dict, Final, List, Tuple, Union

import numpy as np
import yaml

import py123d.conversion.datasets.nuplan.utils as nuplan_utils
from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.nuplan.nuplan_map_conversion import write_nuplan_map
from py123d.conversion.datasets.nuplan.utils.nuplan_constants import (
    NUPLAN_DATA_SPLITS,
    NUPLAN_DEFAULT_DT,
    NUPLAN_LIDAR_DICT,
    NUPLAN_MAP_LOCATIONS,
    NUPLAN_ROLLING_SHUTTER_S,
    NUPLAN_TRAFFIC_STATUS_DICT,
)
from py123d.conversion.datasets.nuplan.utils.nuplan_sql_helper import (
    get_box_detections_for_lidarpc_token_from_db,
    get_nearest_ego_pose_for_timestamp_from_db,
)
from py123d.conversion.log_writer.abstract_log_writer import AbstractLogWriter, CameraData, LiDARData
from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.registry import NuPlanBoxDetectionLabel, NuPlanLiDARIndex
from py123d.datatypes.detections import (
    BoxDetectionSE3,
    BoxDetectionWrapper,
    TrafficLightDetection,
    TrafficLightDetectionWrapper,
)
from py123d.datatypes.metadata import LogMetadata, MapMetadata
from py123d.datatypes.sensors import (
    LiDARMetadata,
    LiDARType,
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from py123d.datatypes.time import TimePoint
from py123d.datatypes.vehicle_state import DynamicStateSE3, EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import get_nuplan_chrysler_pacifica_parameters
from py123d.geometry import PoseSE3, Vector3D

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras, get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.simulation.observation.observation_type import CameraChannel

# NOTE: Leaving this constant here, to avoid having a nuplan dependency in nuplan_constants.py
NUPLAN_CAMERA_MAPPING = {
    PinholeCameraType.PCAM_F0: CameraChannel.CAM_F0,
    PinholeCameraType.PCAM_B0: CameraChannel.CAM_B0,
    PinholeCameraType.PCAM_L0: CameraChannel.CAM_L0,
    PinholeCameraType.PCAM_L1: CameraChannel.CAM_L1,
    PinholeCameraType.PCAM_L2: CameraChannel.CAM_L2,
    PinholeCameraType.PCAM_R0: CameraChannel.CAM_R0,
    PinholeCameraType.PCAM_R1: CameraChannel.CAM_R1,
    PinholeCameraType.PCAM_R2: CameraChannel.CAM_R2,
}

TARGET_DT: Final[float] = 0.1  # TODO: make configurable


def create_splits_logs() -> Dict[str, List[str]]:
    # NOTE: nuPlan stores the training and validataion logs
    yaml_filepath = Path(nuplan_utils.__path__[0]) / "log_splits.yaml"
    with open(yaml_filepath, "r") as stream:
        splits = yaml.safe_load(stream)

    return splits["log_splits"]


class NuPlanConverter(AbstractDatasetConverter):
    """Converter class for the nuPlan dataset."""

    def __init__(
        self,
        splits: List[str],
        nuplan_data_root: Union[Path, str],
        nuplan_maps_root: Union[Path, str],
        nuplan_sensor_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        """Initializes the NuPlanConverter.

        :param splits: List of splits to convert, i.e., ["nuplan_train", "nuplan_val", "nuplan_test"]
        :param nuplan_data_root: Root directory of the nuPlan data.
        :param nuplan_maps_root: Root directory of the nuPlan maps.
        :param nuplan_sensor_root: Root directory of the nuPlan sensor data.
        :param dataset_converter_config: Configuration for the dataset converter.
        """

        super().__init__(dataset_converter_config)
        assert nuplan_data_root is not None, "The variable `nuplan_data_root` must be provided."
        assert nuplan_maps_root is not None, "The variable `nuplan_maps_root` must be provided."
        assert nuplan_sensor_root is not None, "The variable `nuplan_sensor_root` must be provided."
        for split in splits:
            assert (
                split in NUPLAN_DATA_SPLITS
            ), f"Split {split} is not available. Available splits: {NUPLAN_DATA_SPLITS}"

        self._splits: List[str] = splits
        self._nuplan_data_root: Path = Path(nuplan_data_root)
        self._nuplan_maps_root: Path = Path(nuplan_maps_root)
        self._nuplan_sensor_root: Path = Path(nuplan_sensor_root)

        self._split_log_path_pairs: List[Tuple[str, Path]] = self._collect_split_log_path_pairs()

    def _collect_split_log_path_pairs(self) -> List[Tuple[str, Path]]:
        """Collects the (split, log_path) pairs for the specified splits."""

        # NOTE: the nuplan mini folder has an internal train, val, test structure, all stored in "mini".
        # The complete dataset is saved in the "trainval" folder (train and val), or in the "test" folder (for test).
        # Thus, we need filter the logs in a split, based on the internal nuPlan configuration.
        split_log_path_pairs: List[Tuple[str, Path]] = []
        log_names_per_split = create_splits_logs()

        for split in self._splits:
            split_type = split.split("_")[-1]
            assert split_type in ["train", "val", "test"]

            if split in ["nuplan_train", "nuplan_val"]:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "trainval"
            elif split in ["nuplan_test"]:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "test"
            elif split in ["nuplan-mini_train", "nuplan-mini_val", "nuplan-mini_test"]:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "mini"
            else:
                raise ValueError(f"Unknown nuPlan split: {split}")

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

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        location = NUPLAN_MAP_LOCATIONS[map_index]

        # Dummy log metadata for map writing, TODO: Consider using MapMetadata instead?
        map_metadata = _get_nuplan_map_metadata(location)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)
        if map_needs_writing:
            write_nuplan_map(
                location=location,
                nuplan_maps_root=self._nuplan_maps_root,
                map_writer=map_writer,
            )

        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        split, source_log_path = self._split_log_path_pairs[log_index]
        nuplan_log_db = NuPlanDB(str(self._nuplan_data_root), str(source_log_path), None)
        log_name = nuplan_log_db.log_name

        # 1. Initialize log metadata
        log_metadata = LogMetadata(
            dataset="nuplan",
            split=split,
            log_name=log_name,
            location=nuplan_log_db.log.map_version,
            timestep_seconds=TARGET_DT,
            vehicle_parameters=get_nuplan_chrysler_pacifica_parameters(),
            box_detection_label_class=NuPlanBoxDetectionLabel,
            pinhole_camera_metadata=_get_nuplan_camera_metadata(
                source_log_path,
                self._nuplan_sensor_root,
                self.dataset_converter_config,
            ),
            lidar_metadata=_get_nuplan_lidar_metadata(
                self._nuplan_sensor_root,
                log_name,
                self.dataset_converter_config,
            ),
            map_metadata=_get_nuplan_map_metadata(nuplan_log_db.log.map_version),
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        if log_needs_writing:
            step_interval: float = int(TARGET_DT / NUPLAN_DEFAULT_DT)
            for nuplan_lidar_pc in nuplan_log_db.lidar_pc[::step_interval]:

                lidar_pc_token: str = nuplan_lidar_pc.token
                log_writer.write(
                    timestamp=TimePoint.from_us(nuplan_lidar_pc.timestamp),
                    ego_state=_extract_nuplan_ego_state(nuplan_lidar_pc),
                    box_detections=_extract_nuplan_box_detections(nuplan_lidar_pc, source_log_path),
                    traffic_lights=_extract_nuplan_traffic_lights(nuplan_log_db, lidar_pc_token),
                    pinhole_cameras=_extract_nuplan_cameras(
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


def _get_nuplan_map_metadata(location: str) -> MapMetadata:
    """Gets the nuPlan map metadata for a given location."""
    return MapMetadata(
        dataset="nuplan",
        split=None,
        log_name=None,
        location=location,
        map_has_z=False,
        map_is_local=False,
    )


def _get_nuplan_camera_metadata(
    source_log_path: Path,
    nuplan_sensor_root: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, PinholeCameraMetadata]:
    """Extracts the nuPlan camera metadata for a given log."""

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
    if dataset_converter_config.include_pinhole_cameras:
        log_name = source_log_path.stem
        for camera_type, nuplan_camera_type in NUPLAN_CAMERA_MAPPING.items():
            camera_folder = nuplan_sensor_root / log_name / f"{nuplan_camera_type.value}"
            if camera_folder.exists() and camera_folder.is_dir():
                camera_metadata[camera_type] = _get_camera_metadata(camera_type)

    return camera_metadata


def _get_nuplan_lidar_metadata(
    nuplan_sensor_root: Path,
    log_name: str,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LiDARType, LiDARMetadata]:
    """Extracts the nuPlan LiDAR metadata for a given log."""
    metadata: Dict[LiDARType, LiDARMetadata] = {}
    log_lidar_folder = nuplan_sensor_root / log_name / "MergedPointCloud"
    # NOTE: We first need to check if the LiDAR folder exists, as not all logs have LiDAR data
    if log_lidar_folder.exists() and log_lidar_folder.is_dir() and dataset_converter_config.include_lidars:
        for lidar_type in NUPLAN_LIDAR_DICT.values():
            metadata[lidar_type] = LiDARMetadata(
                lidar_type=lidar_type,
                lidar_index=NuPlanLiDARIndex,
                extrinsic=None,  # NOTE: LiDAR extrinsic are unknown
            )
    return metadata


def _extract_nuplan_ego_state(nuplan_lidar_pc: LidarPc) -> EgoStateSE3:
    """Extracts the nuPlan ego state from a given LidarPc database objects."""

    vehicle_parameters = get_nuplan_chrysler_pacifica_parameters()
    rear_axle_pose = PoseSE3(
        x=nuplan_lidar_pc.ego_pose.x,
        y=nuplan_lidar_pc.ego_pose.y,
        z=nuplan_lidar_pc.ego_pose.z,
        qw=nuplan_lidar_pc.ego_pose.qw,
        qx=nuplan_lidar_pc.ego_pose.qx,
        qy=nuplan_lidar_pc.ego_pose.qy,
        qz=nuplan_lidar_pc.ego_pose.qz,
    )
    dynamic_state_se3 = DynamicStateSE3(
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
    return EgoStateSE3.from_rear_axle(
        rear_axle_se3=rear_axle_pose,
        vehicle_parameters=vehicle_parameters,
        dynamic_state_se3=dynamic_state_se3,
    )


def _extract_nuplan_box_detections(lidar_pc: LidarPc, source_log_path: Path) -> BoxDetectionWrapper:
    """Extracts the nuPlan box detections from a given LidarPc database objects."""
    box_detections: List[BoxDetectionSE3] = get_box_detections_for_lidarpc_token_from_db(
        str(source_log_path), lidar_pc.token
    )
    return BoxDetectionWrapper(box_detections=box_detections)


def _extract_nuplan_traffic_lights(log_db: NuPlanDB, lidar_pc_token: str) -> TrafficLightDetectionWrapper:
    """Extracts the nuPlan traffic light detections from a given LidarPc database objects."""
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
) -> List[CameraData]:
    """Extracts the nuPlan camera data from a given LidarPc database objects."""
    camera_data_list: List[CameraData] = []
    if dataset_converter_config.include_pinhole_cameras:
        log_cam_infos = {camera.token: camera for camera in nuplan_log_db.log.cameras}
        for camera_type, camera_channel in NUPLAN_CAMERA_MAPPING.items():
            image_class = list(
                get_images_from_lidar_tokens(str(source_log_path), [nuplan_lidar_pc.token], [str(camera_channel.value)])
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
                    extrinsic = PoseSE3.from_transformation_matrix(c2e)

                    # Store in dictionary
                    camera_data_list.append(
                        CameraData(
                            camera_type=camera_type,
                            extrinsic=extrinsic,
                            dataset_root=nuplan_sensor_root,
                            relative_path=filename_jpg.relative_to(nuplan_sensor_root),
                        )
                    )
    return camera_data_list


def _extract_nuplan_lidars(
    nuplan_lidar_pc: LidarPc,
    nuplan_sensor_root: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> List[LiDARData]:
    """Extracts the nuPlan LiDAR data from a given LidarPc database objects."""
    lidars: List[LiDARData] = []
    if dataset_converter_config.include_lidars:
        lidar_full_path = nuplan_sensor_root / nuplan_lidar_pc.filename
        if lidar_full_path.exists() and lidar_full_path.is_file():
            lidars.append(
                LiDARData(
                    lidar_type=LiDARType.LIDAR_MERGED,
                    dataset_root=nuplan_sensor_root,
                    relative_path=nuplan_lidar_pc.filename,
                )
            )
    return lidars


def _extract_nuplan_scenario_tag(nuplan_log_db: NuPlanDB, lidar_pc_token: str) -> List[str]:
    """Extracts the nuPlan scenario tags from a given LidarPc database objects."""
    scenario_tags = [
        scenario_tag.type for scenario_tag in nuplan_log_db.scenario_tag.select_many(lidar_pc_token=lidar_pc_token)
    ]
    if len(scenario_tags) == 0:
        scenario_tags = ["unknown"]
    return scenario_tags


def _extract_nuplan_route_lane_group_ids(nuplan_lidar_pc: LidarPc) -> List[int]:
    """Extracts the nuPlan route lane group IDs from a given LidarPc database objects."""
    return [
        int(roadblock_id)
        for roadblock_id in str(nuplan_lidar_pc.scene.roadblock_ids).split(" ")
        if len(roadblock_id) > 0
    ]
