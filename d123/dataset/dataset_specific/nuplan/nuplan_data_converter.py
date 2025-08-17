import gc
import json
import os
import pickle
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import yaml
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras, get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.simulation.observation.observation_type import CameraChannel
from pyquaternion import Quaternion
from sqlalchemy import func

import d123.dataset.dataset_specific.nuplan.utils as nuplan_utils
from d123.common.datatypes.detection.detection import TrafficLightStatus
from d123.common.datatypes.detection.detection_types import DetectionType
from d123.common.datatypes.sensor.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar import LiDARMetadata, LiDARType, lidar_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar_index import NuplanLidarIndex
from d123.common.datatypes.time.time_point import TimePoint
from d123.common.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3, EgoStateSE3Index
from d123.common.datatypes.vehicle_state.vehicle_parameters import (
    get_nuplan_pacifica_parameters,
    rear_axle_se3_to_center_se3,
)
from d123.common.geometry.base import StateSE3
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3, BoundingBoxSE3Index
from d123.common.geometry.constants import DEFAULT_PITCH, DEFAULT_ROLL
from d123.common.geometry.vector import Vector3D, Vector3DIndex
from d123.common.multithreading.worker_utils import WorkerPool, worker_map
from d123.dataset.arrow.helper import open_arrow_table, write_arrow_table
from d123.dataset.dataset_specific.nuplan.nuplan_map_conversion import MAP_LOCATIONS, NuPlanMapConverter
from d123.dataset.dataset_specific.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.dataset.logs.log_metadata import LogMetadata

TARGET_DT: Final[float] = 0.1
NUPLAN_DT: Final[float] = 0.05
SORT_BY_TIMESTAMP: Final[bool] = True

NUPLAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "red": TrafficLightStatus.RED,
    "unknown": TrafficLightStatus.UNKNOWN,
}
NUPLAN_DETECTION_NAME_DICT = {
    "vehicle": DetectionType.VEHICLE,
    "bicycle": DetectionType.BICYCLE,
    "pedestrian": DetectionType.PEDESTRIAN,
    "traffic_cone": DetectionType.TRAFFIC_CONE,
    "barrier": DetectionType.BARRIER,
    "czone_sign": DetectionType.CZONE_SIGN,
    "generic_object": DetectionType.GENERIC_OBJECT,
}

NUPLAN_CAMERA_TYPES = {
    CameraType.CAM_F0: CameraChannel.CAM_F0,
    CameraType.CAM_B0: CameraChannel.CAM_B0,
    CameraType.CAM_L0: CameraChannel.CAM_L0,
    CameraType.CAM_L1: CameraChannel.CAM_L1,
    CameraType.CAM_L2: CameraChannel.CAM_L2,
    CameraType.CAM_R0: CameraChannel.CAM_R0,
    CameraType.CAM_R1: CameraChannel.CAM_R1,
    CameraType.CAM_R2: CameraChannel.CAM_R2,
}

NUPLAN_DATA_ROOT = Path(os.environ["NUPLAN_DATA_ROOT"])
NUPLAN_ROLLING_SHUTTER_S: Final[TimePoint] = TimePoint.from_s(1 / 60)


def create_splits_logs() -> Dict[str, List[str]]:
    yaml_filepath = Path(nuplan_utils.__path__[0]) / "log_splits.yaml"
    with open(yaml_filepath, "r") as stream:
        splits = yaml.safe_load(stream)

    return splits["log_splits"]


class NuplanDataConverter(RawDataConverter):
    def __init__(
        self,
        splits: List[str],
        log_path: Union[Path, str],
        data_converter_config: DataConverterConfig,
    ) -> None:
        super().__init__(data_converter_config)
        for split in splits:
            assert (
                split in self.get_available_splits()
            ), f"Split {split} is not available. Available splits: {self.available_splits}"

        self._splits: List[str] = splits
        self._log_path: Path = Path(log_path)
        self._log_paths_per_split: Dict[str, List[Path]] = self._collect_log_paths()
        self._target_dt: float = 0.1

    def _collect_log_paths(self) -> Dict[str, List[Path]]:
        # NOTE: the nuplan mini folder has an internal train, val, test structure, all stored in "mini".
        # The complete dataset is saved in the "trainval" folder (train and val), or in the "test" folder (for test).
        subsplit_log_names: Dict[str, List[str]] = create_splits_logs()
        log_paths_per_split: Dict[str, List[Path]] = {}

        for split in self._splits:
            subsplit = split.split("_")[-1]
            assert subsplit in ["train", "val", "test"]
            if split in ["nuplan_train", "nuplan_val"]:
                log_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / "trainval"
            elif split in ["nuplan_test"]:
                log_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / "test"
            elif split in ["nuplan_mini_train", "nuplan_mini_val", "nuplan_mini_test"]:
                log_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / "mini"
            elif split == "nuplan_private_test":
                log_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / "private_test"

            all_log_files_in_path = [log_file for log_file in log_path.glob("*.db")]
            all_log_names = set([str(log_file.stem) for log_file in all_log_files_in_path])
            set(subsplit_log_names[subsplit])
            # log_paths = [log_path / f"{log_name}.db" for log_name in list(all_log_names & split_log_names)]
            log_paths = [log_path / f"{log_name}.db" for log_name in list(all_log_names)]
            log_paths_per_split[split] = log_paths

        return log_paths_per_split

    def get_available_splits(self) -> List[str]:
        return [
            "nuplan_train",
            "nuplan_val",
            "nuplan_test",
            "nuplan_mini_train",
            "nuplan_mini_val",
            "nuplan_mini_test",
            "nuplan_private_test",
        ]

    def convert_maps(self, worker: WorkerPool) -> None:
        worker_map(
            worker,
            partial(convert_nuplan_map_to_gpkg, data_converter_config=self.data_converter_config),
            list(MAP_LOCATIONS),
        )

    def convert_logs(self, worker: WorkerPool) -> None:
        log_args = [
            {
                "log_path": log_path,
                "split": split,
            }
            for split, log_paths in self._log_paths_per_split.items()
            for log_path in log_paths
        ]

        worker_map(
            worker,
            partial(
                convert_nuplan_log_to_arrow,
                data_converter_config=self.data_converter_config,
            ),
            log_args,
        )


def convert_nuplan_map_to_gpkg(map_names: List[str], data_converter_config: DataConverterConfig) -> List[Any]:
    for map_name in map_names:
        map_path = data_converter_config.output_path / "maps" / f"nuplan_{map_name}.gpkg"
        if data_converter_config.force_map_conversion or not map_path.exists():
            map_path.unlink(missing_ok=True)
            NuPlanMapConverter(data_converter_config.output_path / "maps").convert(map_name=map_name)
    return []


def convert_nuplan_log_to_arrow(
    args: List[Dict[str, Union[List[str], List[Path]]]], data_converter_config: DataConverterConfig
) -> List[Any]:
    for log_info in args:
        log_path: Path = log_info["log_path"]
        split: str = log_info["split"]

        if not log_path.exists():
            raise FileNotFoundError(f"Log path {log_path} does not exist.")

        log_db = NuPlanDB(NUPLAN_DATA_ROOT, str(log_path), None)
        log_file_path = data_converter_config.output_path / split / f"{log_path.stem}.arrow"

        if data_converter_config.force_log_conversion or not log_file_path.exists():
            log_file_path.unlink(missing_ok=True)
            if not log_file_path.parent.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            metadata = LogMetadata(
                dataset="nuplan",
                log_name=log_db.log_name,
                location=log_db.log.map_version,
                timestep_seconds=TARGET_DT,
                map_has_z=False,
            )
            vehicle_parameters = get_nuplan_pacifica_parameters()
            camera_metadata = get_nuplan_camera_metadata(log_path)
            lidar_metadata = get_nuplan_lidar_metadata(log_db)

            schema_column_list = [
                ("token", pa.string()),
                ("timestamp", pa.int64()),
                ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                ("detections_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                ("detections_token", pa.list_(pa.string())),
                ("detections_type", pa.list_(pa.int16())),
                ("ego_states", pa.list_(pa.float64(), len(EgoStateSE3Index))),
                ("traffic_light_ids", pa.list_(pa.int64())),
                ("traffic_light_types", pa.list_(pa.int16())),
                ("scenario_tag", pa.list_(pa.string())),
                ("route_lane_group_ids", pa.list_(pa.int64())),
            ]
            if data_converter_config.lidar_store_option is not None:
                for lidar_type in lidar_metadata.keys():
                    if data_converter_config.lidar_store_option == "path":
                        schema_column_list.append((lidar_type.serialize(), pa.string()))
                    elif data_converter_config.lidar_store_option == "binary":
                        raise NotImplementedError("Binary lidar storage is not implemented.")

            if data_converter_config.camera_store_option is not None:
                for camera_type in camera_metadata.keys():
                    if data_converter_config.camera_store_option == "path":
                        schema_column_list.append((camera_type.serialize(), pa.string()))
                        schema_column_list.append(
                            (f"{camera_type.serialize()}_extrinsic", pa.list_(pa.float64(), 4 * 4))
                        )

                    elif data_converter_config.camera_store_option == "binary":
                        raise NotImplementedError("Binary camera storage is not implemented.")

            recording_schema = pa.schema(schema_column_list)
            recording_schema = recording_schema.with_metadata(
                {
                    "log_metadata": json.dumps(asdict(metadata)),
                    "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                    "camera_metadata": camera_metadata_dict_to_json(camera_metadata),
                    "lidar_metadata": lidar_metadata_dict_to_json(lidar_metadata),
                }
            )

            _write_recording_table(log_db, recording_schema, log_file_path, log_path, data_converter_config)

            log_db.detach_tables()
            log_db.remove_ref()
            del recording_schema, vehicle_parameters, log_db
        gc.collect()
    return []


def get_nuplan_camera_metadata(log_path: Path) -> Dict[CameraType, CameraMetadata]:

    def _get_camera_metadata(camera_type: CameraType) -> CameraMetadata:
        cam = list(get_cameras(log_path, [str(NUPLAN_CAMERA_TYPES[camera_type].value)]))[0]
        intrinsic = np.array(pickle.loads(cam.intrinsic))
        rotation = np.array(pickle.loads(cam.rotation))
        rotation = Quaternion(rotation).rotation_matrix
        distortion = np.array(pickle.loads(cam.distortion))
        return CameraMetadata(
            camera_type=camera_type,
            width=cam.width,
            height=cam.height,
            intrinsic=intrinsic,
            distortion=distortion,
        )

    log_cam_infos: Dict[str, CameraMetadata] = {}
    for camera_type in NUPLAN_CAMERA_TYPES.keys():
        log_cam_infos[camera_type] = _get_camera_metadata(camera_type)

    return log_cam_infos


def get_nuplan_lidar_metadata(log_db: NuPlanDB) -> Dict[LiDARType, LiDARMetadata]:
    metadata: Dict[LiDARType, LiDARMetadata] = {}
    metadata[LiDARType.LIDAR_MERGED] = LiDARMetadata(
        lidar_type=LiDARType.LIDAR_MERGED,
        lidar_index=NuplanLidarIndex,
        extrinsic=None,  # NOTE: LiDAR extrinsic are unknown
    )
    return metadata


def _write_recording_table(
    log_db: NuPlanDB,
    recording_schema: pa.schema,
    log_file_path: Path,
    source_log_path: Path,
    data_converter_config: DataConverterConfig,
) -> None:

    # with pa.ipc.new_stream(str(log_file_path), recording_schema) as writer:
    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:
            step_interval: float = int(TARGET_DT / NUPLAN_DT)
            for lidar_pc in log_db.lidar_pc[::step_interval]:
                lidar_pc_token: str = lidar_pc.token
                (
                    detections_state,
                    detections_velocity,
                    detections_token,
                    detections_types,
                ) = _extract_detections(lidar_pc)
                traffic_light_ids, traffic_light_types = _extract_traffic_lights(log_db, lidar_pc_token)
                route_lane_group_ids = [
                    int(roadblock_id)
                    for roadblock_id in str(lidar_pc.scene.roadblock_ids).split(" ")
                    if len(roadblock_id) > 0
                ]

                row_data = {
                    "token": [lidar_pc_token],
                    "timestamp": [lidar_pc.timestamp],
                    "detections_state": [detections_state],
                    "detections_velocity": [detections_velocity],
                    "detections_token": [detections_token],
                    "detections_type": [detections_types],
                    "ego_states": [_extract_ego_state(lidar_pc)],
                    "traffic_light_ids": [traffic_light_ids],
                    "traffic_light_types": [traffic_light_types],
                    "scenario_tag": [_extract_scenario_tag(log_db, lidar_pc_token)],
                    "route_lane_group_ids": [route_lane_group_ids],
                }

                if data_converter_config.lidar_store_option is not None:
                    lidar_data_dict = _extract_lidar(lidar_pc, data_converter_config)
                    for lidar_type, lidar_data in lidar_data_dict.items():
                        if lidar_data is not None:
                            row_data[lidar_type.serialize()] = [lidar_data]
                        else:
                            row_data[lidar_type.serialize()] = [None]

                if data_converter_config.camera_store_option is not None:
                    camera_data_dict = _extract_camera(log_db, lidar_pc, source_log_path, data_converter_config)
                    for camera_type, camera_data in camera_data_dict.items():
                        if camera_data is not None:
                            row_data[camera_type.serialize()] = [camera_data[0]]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [camera_data[1]]
                        else:
                            row_data[camera_type.serialize()] = [None]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [None]

                batch = pa.record_batch(row_data, schema=recording_schema)
                writer.write_batch(batch)
                del batch, row_data, detections_state, detections_velocity, detections_token, detections_types

    if SORT_BY_TIMESTAMP:
        recording_table = open_arrow_table(log_file_path)
        recording_table = recording_table.sort_by([("timestamp", "ascending")])
        write_arrow_table(recording_table, log_file_path)


def _extract_detections(lidar_pc: LidarPc) -> Tuple[List[List[float]], List[List[float]], List[str], List[int]]:
    detections_state: List[List[float]] = []
    detections_velocity: List[List[float]] = []
    detections_token: List[str] = []
    detections_types: List[int] = []

    for lidar_box in lidar_pc.lidar_boxes:
        lidar_box: LidarBox
        center = StateSE3(
            x=lidar_box.x,
            y=lidar_box.y,
            z=lidar_box.z,
            roll=DEFAULT_ROLL,
            pitch=DEFAULT_PITCH,
            yaw=lidar_box.yaw,
        )
        bounding_box_se3 = BoundingBoxSE3(center, lidar_box.length, lidar_box.width, lidar_box.height)

        detections_state.append(bounding_box_se3.array)
        detections_velocity.append(lidar_box.velocity)
        detections_token.append(lidar_box.track_token)
        detections_types.append(int(NUPLAN_DETECTION_NAME_DICT[lidar_box.category.name]))

    return detections_state, detections_velocity, detections_token, detections_types


def _extract_ego_state(lidar_pc: LidarPc) -> List[float]:

    yaw, pitch, roll = lidar_pc.ego_pose.quaternion.yaw_pitch_roll
    vehicle_parameters = get_nuplan_pacifica_parameters()
    # vehicle_parameters = get_pacifica_parameters()

    rear_axle_pose = StateSE3(
        x=lidar_pc.ego_pose.x,
        y=lidar_pc.ego_pose.y,
        z=lidar_pc.ego_pose.z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )
    # NOTE: The height to rear axle is not provided the dataset and is merely approximated.
    center = rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_pose, vehicle_parameters=vehicle_parameters)
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(
            x=lidar_pc.ego_pose.vx,
            y=lidar_pc.ego_pose.vy,
            z=lidar_pc.ego_pose.vz,
        ),
        acceleration=Vector3D(
            x=lidar_pc.ego_pose.acceleration_x,
            y=lidar_pc.ego_pose.acceleration_y,
            z=lidar_pc.ego_pose.acceleration_z,
        ),
        angular_velocity=Vector3D(
            x=lidar_pc.ego_pose.angular_rate_x,
            y=lidar_pc.ego_pose.angular_rate_y,
            z=lidar_pc.ego_pose.angular_rate_z,
        ),
    )

    return EgoStateSE3(
        center_se3=center,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,
    ).array.tolist()


def _extract_traffic_lights(log_db: NuPlanDB, lidar_pc_token: str) -> Tuple[List[int], List[int]]:
    traffic_light_ids: List[int] = []
    traffic_light_types: List[int] = []
    traffic_lights = log_db.traffic_light_status.select_many(lidar_pc_token=lidar_pc_token)
    for traffic_light in traffic_lights:
        traffic_light_ids.append(int(traffic_light.lane_connector_id))
        traffic_light_types.append(int(NUPLAN_TRAFFIC_STATUS_DICT[traffic_light.status].value))
    return traffic_light_ids, traffic_light_types


def _extract_scenario_tag(log_db: NuPlanDB, lidar_pc_token: str) -> List[str]:

    scenario_tags = [
        scenario_tag.type for scenario_tag in log_db.scenario_tag.select_many(lidar_pc_token=lidar_pc_token)
    ]
    if len(scenario_tags) == 0:
        scenario_tags = ["unknown"]
    return scenario_tags


def _extract_camera(
    log_db: NuPlanDB,
    lidar_pc: LidarPc,
    source_log_path: Path,
    data_converter_config: DataConverterConfig,
) -> Dict[CameraType, Union[str, bytes]]:

    camera_dict: Dict[str, Union[str, bytes]] = {}
    sensor_root = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "sensor_blobs"

    log_cam_infos = {camera.token: camera for camera in log_db.log.cameras}

    for camera_type, camera_channel in NUPLAN_CAMERA_TYPES.items():
        camera_data: Optional[Union[str, bytes]] = None
        c2e: Optional[List[float]] = None
        image_class = list(get_images_from_lidar_tokens(source_log_path, [lidar_pc.token], [str(camera_channel.value)]))
        if len(image_class) != 0:
            image = image_class[0]
            filename_jpg = sensor_root / image.filename_jpg
            if filename_jpg.exists():

                # Code taken from MTGS
                # https://github.com/OpenDriveLab/MTGS/blob/main/nuplan_scripts/utils/nuplan_utils_custom.py#L117

                timestamp = image.timestamp + NUPLAN_ROLLING_SHUTTER_S.time_us
                img_ego_pose: EgoPose = (
                    log_db.log._session.query(EgoPose).order_by(func.abs(EgoPose.timestamp - timestamp)).first()
                )
                img_e2g = img_ego_pose.trans_matrix
                g2e = lidar_pc.ego_pose.trans_matrix_inv
                img_e2e = g2e @ img_e2g
                cam_info = log_cam_infos[image.camera_token]
                c2img_e = cam_info.trans_matrix
                c2e = img_e2e @ c2img_e

                if data_converter_config.camera_store_option == "path":
                    camera_data = str(filename_jpg), c2e.flatten().tolist()
                elif data_converter_config.camera_store_option == "binary":
                    with open(filename_jpg, "rb") as f:
                        camera_data = f.read(), c2e

        camera_dict[camera_type] = camera_data

    return camera_dict


def _extract_lidar(lidar_pc: LidarPc, data_converter_config: DataConverterConfig) -> Dict[LiDARType, Optional[str]]:

    lidar: Optional[str] = None
    lidar_full_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "sensor_blobs" / lidar_pc.filename
    if lidar_full_path.exists():
        lidar = lidar_pc.filename

    return {LiDARType.LIDAR_MERGED: lidar}
