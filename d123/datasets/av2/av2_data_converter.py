import gc
import hashlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pyquaternion import Quaternion

from d123.common.multithreading.worker_utils import WorkerPool, worker_map
from d123.datasets.av2.av2_constants import (
    AV2_CAMERA_TYPE_MAPPING,
    AV2_TO_DETECTION_TYPE,
    AV2SensorBoxDetectionType,
)
from d123.datasets.av2.av2_helper import (
    build_sensor_dataframe,
    build_synchronization_dataframe,
    find_closest_target_fpath,
    get_slice_with_timestamp_ns,
)
from d123.datasets.av2.av2_map_conversion import convert_av2_map
from d123.datasets.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.datatypes.sensors.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.datatypes.sensors.lidar import LiDARMetadata, LiDARType, lidar_metadata_dict_to_json
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3, EgoStateSE3Index
from d123.datatypes.vehicle_state.vehicle_parameters import (
    get_av2_ford_fusion_hybrid_parameters,
    rear_axle_se3_to_center_se3,
)
from d123.geometry import BoundingBoxSE3Index, EulerStateSE3, Vector3D, Vector3DIndex
from d123.geometry.transform.transform_euler_se3 import (
    convert_relative_to_absolute_euler_se3_array,
    get_rotation_matrix,
)
from d123.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL


def create_token(input_data: str) -> str:
    # TODO: Refactor this function.
    # TODO: Add a general function to create tokens from arbitrary data.
    if isinstance(input_data, str):
        input_data = input_data.encode("utf-8")

    hash_obj = hashlib.sha256(input_data)
    return hash_obj.hexdigest()[:16]


class AV2SensorDataConverter(RawDataConverter):
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
        self._data_root: Path = Path(log_path)
        self._log_paths_per_split: Dict[str, List[Path]] = self._collect_log_paths()
        self._target_dt: float = 0.1

    def _collect_log_paths(self) -> Dict[str, List[Path]]:
        log_paths_per_split: Dict[str, List[Path]] = {}

        for split in self._splits:
            subsplit = split.split("_")[-1]
            assert subsplit in ["train", "val", "test"]

            if "av2_sensor" in split:
                log_folder = self._data_root / "sensor" / subsplit
            elif "av2_lidar" in split:
                log_folder = self._data_root / "lidar" / subsplit
            elif "av2_motion" in split:
                log_folder = self._data_root / "motion-forecasting" / subsplit
            elif "av2-sensor-mini" in split:
                log_folder = self._data_root / "sensor_mini" / subsplit

            log_paths_per_split[split] = list(log_folder.iterdir())

        return log_paths_per_split

    def get_available_splits(self) -> List[str]:
        return [
            "av2-sensor_train",
            "av2-sensor_val",
            "av2-sensor_test",
            "av2-sensor-mini_train",
            "av2-sensor-mini_val",
            "av2-sensor-mini_test",
        ]

    def convert_maps(self, worker: WorkerPool) -> None:
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
            partial(convert_av2_map_to_gpkg, data_converter_config=self.data_converter_config),
            log_args,
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
                convert_av2_log_to_arrow,
                data_converter_config=self.data_converter_config,
            ),
            log_args,
        )


def convert_av2_map_to_gpkg(
    args: List[Dict[str, Union[List[str], List[Path]]]],
    data_converter_config: DataConverterConfig,
) -> List[Any]:
    for log_info in args:
        source_log_path: Path = log_info["log_path"]
        split: str = log_info["split"]

        source_log_name = source_log_path.name

        map_path = data_converter_config.output_path / "maps" / split / f"{source_log_name}.gpkg"
        if data_converter_config.force_map_conversion or not map_path.exists():
            map_path.unlink(missing_ok=True)
            convert_av2_map(source_log_path, map_path)
    return []


def convert_av2_log_to_arrow(
    args: List[Dict[str, Union[List[str], List[Path]]]],
    data_converter_config: DataConverterConfig,
) -> List[Any]:
    for log_info in args:
        log_path: Path = log_info["log_path"]
        split: str = log_info["split"]

        if not log_path.exists():
            raise FileNotFoundError(f"Log path {log_path} does not exist.")

        log_file_path = data_converter_config.output_path / split / f"{log_path.stem}.arrow"

        if data_converter_config.force_log_conversion or not log_file_path.exists():
            log_file_path.unlink(missing_ok=True)
            if not log_file_path.parent.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            sensor_df = build_sensor_dataframe(log_path)
            synchronization_df = build_synchronization_dataframe(sensor_df)

            metadata = LogMetadata(
                dataset="av2-sensor",
                log_name=log_path.name,
                location=None,
                timestep_seconds=0.1,  # TODO: verify this
                map_has_z=True,
            )
            vehicle_parameters = get_av2_ford_fusion_hybrid_parameters()  # TODO: Add av2 vehicle parameters
            camera_metadata = get_av2_camera_metadata(log_path)
            lidar_metadata = get_av2_lidar_metadata(log_path)

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

                    elif data_converter_config.camera_store_option == "binary":
                        schema_column_list.append((camera_type.serialize(), pa.binary()))

                    schema_column_list.append((f"{camera_type.serialize()}_extrinsic", pa.list_(pa.float64(), 4 * 4)))

            recording_schema = pa.schema(schema_column_list)
            recording_schema = recording_schema.with_metadata(
                {
                    "log_metadata": json.dumps(asdict(metadata)),
                    "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                    "camera_metadata": camera_metadata_dict_to_json(camera_metadata),
                    "lidar_metadata": lidar_metadata_dict_to_json(lidar_metadata),
                }
            )

            _write_recording_table(
                sensor_df,
                synchronization_df,
                recording_schema,
                log_file_path,
                log_path,
                data_converter_config,
            )
            del recording_schema, vehicle_parameters
        gc.collect()
    return []


def get_av2_camera_metadata(log_path: Path) -> Dict[CameraType, CameraMetadata]:

    intrinsics_file = log_path / "calibration" / "intrinsics.feather"
    intrinsics_df = pd.read_feather(intrinsics_file)

    camera_metadata: Dict[CameraType, CameraMetadata] = {}
    for _, row in intrinsics_df.iterrows():
        row = row.to_dict()

        camera_type = AV2_CAMERA_TYPE_MAPPING[row["sensor_name"]]
        camera_metadata[camera_type] = CameraMetadata(
            camera_type=camera_type,
            width=row["width_px"],
            height=row["height_px"],
            intrinsic=np.array(
                [[row["fx_px"], 0, row["cx_px"]], [0, row["fy_px"], row["cy_px"]], [0, 0, 1]], dtype=np.float64
            ),
            distortion=np.array([row["k1"], row["k2"], row["k3"], 0, 0], dtype=np.float64),
        )

    return camera_metadata


def get_av2_lidar_metadata(log_path: Path) -> Dict[LiDARType, LiDARMetadata]:
    # metadata: Dict[LiDARType, LiDARMetadata] = {}
    # metadata[LiDARType.LIDAR_MERGED] = LiDARMetadata(
    #     lidar_type=LiDARType.LIDAR_MERGED,
    #     lidar_index=NuplanLidarIndex,
    #     extrinsic=None,  # NOTE: LiDAR extrinsic are unknown
    # )
    # return metadata
    return {}


def _write_recording_table(
    sensor_df: pd.DataFrame,
    synchronization_df: pd.DataFrame,
    recording_schema: pa.schema,
    log_file_path: Path,
    source_log_path: Path,
    data_converter_config: DataConverterConfig,
) -> None:

    # NOTE: Similar to other datasets, we use the lidar timestamps as reference timestamps.
    lidar_sensor = sensor_df.xs(key="lidar", level=2)
    lidar_timestamps_ns = np.sort([int(idx_tuple[2]) for idx_tuple in lidar_sensor.index])

    # NOTE: The annotation dataframe is not available for the test split.
    annotations_df = (
        pd.read_feather(source_log_path / "annotations.feather")
        if (source_log_path / "annotations.feather").exists()
        else None
    )

    city_se3_egovehicle_df = pd.read_feather(source_log_path / "city_SE3_egovehicle.feather")

    egovehicle_se3_sensor_df = (
        pd.read_feather(source_log_path / "calibration" / "egovehicle_SE3_sensor.feather")
        if data_converter_config.camera_store_option is not None
        else None
    )

    # with pa.ipc.new_stream(str(log_file_path), recording_schema) as writer:
    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:

            for lidar_timestamp_ns in lidar_timestamps_ns:

                ego_state_se3 = _extract_ego_state(city_se3_egovehicle_df, lidar_timestamp_ns)
                (
                    detections_state,
                    detections_velocity,
                    detections_token,
                    detections_types,
                ) = _extract_box_detections(annotations_df, lidar_timestamp_ns, ego_state_se3)
                traffic_light_ids, traffic_light_types = _extract_traffic_lights()
                route_lane_group_ids = None  # TODO: Add route lane group ids extraction ?
                row_data = {
                    "token": [create_token(str(lidar_timestamp_ns))],
                    "timestamp": [TimePoint.from_ns(int(lidar_timestamp_ns)).time_us],
                    "detections_state": [detections_state],
                    "detections_velocity": [detections_velocity],
                    "detections_token": [detections_token],
                    "detections_type": [detections_types],
                    "ego_states": [ego_state_se3.array.tolist()],
                    "traffic_light_ids": [traffic_light_ids],
                    "traffic_light_types": [traffic_light_types],
                    "scenario_tag": [_extract_scenario_tag()],
                    "route_lane_group_ids": [route_lane_group_ids],
                }

                # TODO: add lidar data

                # if data_converter_config.lidar_store_option is not None:
                #     lidar_data_dict = _extract_lidar(lidar_pc, data_converter_config)
                #     for lidar_type, lidar_data in lidar_data_dict.items():
                #         if lidar_data is not None:
                #             row_data[lidar_type.serialize()] = [lidar_data]
                #         else:
                #             row_data[lidar_type.serialize()] = [None]

                if data_converter_config.camera_store_option is not None:
                    camera_data_dict = _extract_camera(
                        lidar_timestamp_ns,
                        city_se3_egovehicle_df,
                        egovehicle_se3_sensor_df,
                        ego_state_se3,
                        synchronization_df,
                        source_log_path,
                        data_converter_config,
                    )
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


def _extract_box_detections(
    annotations_df: Optional[pd.DataFrame],
    lidar_timestamp_ns: int,
    ego_state_se3: EgoStateSE3,
) -> Tuple[List[List[float]], List[List[float]], List[str], List[int]]:

    # TODO: Extract velocity from annotations_df if available.

    if annotations_df is None:
        return [], [], [], []

    annotations_slice = get_slice_with_timestamp_ns(annotations_df, lidar_timestamp_ns)
    num_detections = len(annotations_slice)

    detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    detections_token: List[str] = annotations_slice["track_uuid"].tolist()
    detections_types: List[int] = []

    for detection_idx, (_, row) in enumerate(annotations_slice.iterrows()):
        row = row.to_dict()
        yaw, pitch, roll = Quaternion(
            w=row["qw"],
            x=row["qx"],
            y=row["qy"],
            z=row["qz"],
        ).yaw_pitch_roll

        detections_state[detection_idx, BoundingBoxSE3Index.X] = row["tx_m"]
        detections_state[detection_idx, BoundingBoxSE3Index.Y] = row["ty_m"]
        detections_state[detection_idx, BoundingBoxSE3Index.Z] = row["tz_m"]
        detections_state[detection_idx, BoundingBoxSE3Index.ROLL] = roll
        detections_state[detection_idx, BoundingBoxSE3Index.PITCH] = pitch
        detections_state[detection_idx, BoundingBoxSE3Index.YAW] = yaw
        detections_state[detection_idx, BoundingBoxSE3Index.LENGTH] = row["length_m"]
        detections_state[detection_idx, BoundingBoxSE3Index.WIDTH] = row["width_m"]
        detections_state[detection_idx, BoundingBoxSE3Index.HEIGHT] = row["height_m"]

        av2_detection_type = AV2SensorBoxDetectionType.deserialize(row["category"])
        detections_types.append(int(AV2_TO_DETECTION_TYPE[av2_detection_type]))

    detections_state[:, BoundingBoxSE3Index.STATE_SE3] = convert_relative_to_absolute_euler_se3_array(
        origin=ego_state_se3.rear_axle_se3, se3_array=detections_state[:, BoundingBoxSE3Index.STATE_SE3]
    )

    ZERO_BOX_ROLL_PITCH = False  # TODO: Add config option or remove
    if ZERO_BOX_ROLL_PITCH:
        detections_state[:, BoundingBoxSE3Index.ROLL] = DEFAULT_ROLL
        detections_state[:, BoundingBoxSE3Index.PITCH] = DEFAULT_PITCH

    return detections_state.tolist(), detections_velocity.tolist(), detections_token, detections_types


def _extract_ego_state(city_se3_egovehicle_df: pd.DataFrame, lidar_timestamp_ns: int) -> EgoStateSE3:
    ego_state_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, lidar_timestamp_ns)
    assert (
        len(ego_state_slice) == 1
    ), f"Expected exactly one ego state for timestamp {lidar_timestamp_ns}, got {len(ego_state_slice)}."

    ego_pose_dict = ego_state_slice.iloc[0].to_dict()

    ego_pose_quat = Quaternion(
        w=ego_pose_dict["qw"],
        x=ego_pose_dict["qx"],
        y=ego_pose_dict["qy"],
        z=ego_pose_dict["qz"],
    )

    yaw, pitch, roll = ego_pose_quat.yaw_pitch_roll

    rear_axle_pose = EulerStateSE3(
        x=ego_pose_dict["tx_m"],
        y=ego_pose_dict["ty_m"],
        z=ego_pose_dict["tz_m"],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )
    vehicle_parameters = get_av2_ford_fusion_hybrid_parameters()  # TODO: Add av2 vehicle parameters
    center = rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_pose, vehicle_parameters=vehicle_parameters)
    # TODO: Add script to calculate the dynamic state from log sequence.
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        acceleration=Vector3D(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        angular_velocity=Vector3D(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
    )

    return EgoStateSE3(
        center_se3=center,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,
    )


def _extract_traffic_lights() -> Tuple[List[int], List[int]]:
    return [], []


def _extract_scenario_tag() -> List[str]:
    return ["unknown"]


def _extract_camera(
    lidar_timestamp_ns: int,
    city_se3_egovehicle_df: pd.DataFrame,
    egovehicle_se3_sensor_df: pd.DataFrame,
    ego_state_se3: EgoStateSE3,
    synchronization_df: pd.DataFrame,
    source_log_path: Path,
    data_converter_config: DataConverterConfig,
) -> Dict[CameraType, Union[str, bytes]]:

    camera_dict: Dict[CameraType, Union[str, bytes]] = {
        camera_type: None for camera_type in AV2_CAMERA_TYPE_MAPPING.values()
    }
    split = source_log_path.parent.name
    log_id = source_log_path.name

    source_dataset_dir = source_log_path.parent.parent

    rear_axle_se3 = ego_state_se3.rear_axle_se3
    ego_transform = np.zeros((4, 4), dtype=np.float64)
    ego_transform[:3, :3] = get_rotation_matrix(ego_state_se3.rear_axle_se3)
    ego_transform[:3, 3] = rear_axle_se3.point_3d.array

    for _, row in egovehicle_se3_sensor_df.iterrows():
        row = row.to_dict()
        if row["sensor_name"] not in AV2_CAMERA_TYPE_MAPPING:
            continue

        camera_name = row["sensor_name"]
        camera_type = AV2_CAMERA_TYPE_MAPPING[camera_name]

        relative_image_path = find_closest_target_fpath(
            split=split,
            log_id=log_id,
            src_sensor_name="lidar",
            src_timestamp_ns=lidar_timestamp_ns,
            target_sensor_name=camera_name,
            synchronization_df=synchronization_df,
        )
        if relative_image_path is None:
            camera_dict[camera_type] = None
        else:
            absolute_image_path = source_dataset_dir / relative_image_path
            assert absolute_image_path.exists()
            # TODO: Adjust for finer IMU timestamps to correct the camera extrinsic.
            camera_extrinsic = np.eye(4, dtype=np.float64)
            camera_extrinsic[:3, :3] = Quaternion(
                w=row["qw"],
                x=row["qx"],
                y=row["qy"],
                z=row["qz"],
            ).rotation_matrix
            camera_extrinsic[:3, 3] = np.array([row["tx_m"], row["ty_m"], row["tz_m"]], dtype=np.float64)
            # camera_extrinsic = camera_extrinsic @ ego_transform
            camera_extrinsic = camera_extrinsic.flatten().tolist()

            if data_converter_config.camera_store_option == "path":
                camera_dict[camera_type] = (str(relative_image_path), camera_extrinsic)
            elif data_converter_config.camera_store_option == "binary":
                with open(absolute_image_path, "rb") as f:
                    camera_dict[camera_type] = (f.read(), camera_extrinsic)

    return camera_dict


def _extract_lidar(lidar_pc, data_converter_config: DataConverterConfig) -> Dict[LiDARType, Optional[str]]:

    # lidar: Optional[str] = None
    # lidar_full_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "sensor_blobs" / lidar_pc.filename
    # if lidar_full_path.exists():
    #     lidar = lidar_pc.filename

    # return {LiDARType.LIDAR_MERGED: lidar}
    return {}
