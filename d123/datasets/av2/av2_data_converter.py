import gc
import hashlib
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from typing_extensions import Final

from d123.common.multithreading.worker_utils import WorkerPool, worker_map
from d123.common.utils.arrow_helper import open_arrow_table, write_arrow_table
from d123.datasets.av2.av2_constants import AV2_CAMERA_TYPE_MAPPING, AV2_TO_DETECTION_TYPE, AV2SensorBoxDetectionType
from d123.datasets.av2.av2_helper import (
    build_sensor_dataframe,
    build_synchronization_dataframe,
    find_closest_target_fpath,
    get_slice_with_timestamp_ns,
)
from d123.datasets.av2.av2_map_conversion import convert_av2_map
from d123.datasets.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.datasets.utils.arrow_ipc_writer import ArrowLogWriter
from d123.datatypes.detections.detection import BoxDetectionMetadata, BoxDetectionSE3, BoxDetectionWrapper
from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.scene.scene_metadata import LogMetadata
<<<<<<< HEAD
from d123.datatypes.sensors.camera import PinholeCameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.datatypes.sensors.lidar import LiDARMetadata, LiDARType, lidar_metadata_dict_to_json
=======
from d123.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from d123.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
>>>>>>> dev_v0.0.7
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3
from d123.datatypes.vehicle_state.vehicle_parameters import (
    get_av2_ford_fusion_hybrid_parameters,
    rear_axle_se3_to_center_se3,
)
from d123.geometry import BoundingBoxSE3Index, StateSE3, Vector3D, Vector3DIndex
from d123.geometry.bounding_box import BoundingBoxSE3
from d123.geometry.transform.transform_se3 import convert_relative_to_absolute_se3_array

SORT_BY_TIMESTAMP: Final[bool] = True


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

            log_metadata = LogMetadata(
                dataset="av2-sensor",
                split=split,
                log_name=log_path.name,
                location=None,  # TODO: Add location information.
                timestep_seconds=0.1,
                vehicle_parameters=get_av2_ford_fusion_hybrid_parameters(),
                camera_metadata=get_av2_camera_metadata(log_path),
                lidar_metadata=get_av2_lidar_metadata(log_path),
                map_has_z=True,
                map_is_local=True,
            )

            log_writer = ArrowLogWriter(
                log_path=log_file_path,
                data_converter_config=data_converter_config,
                log_metadata=log_metadata,
            )

            _write_recording_table(
                sensor_df,
                synchronization_df,
                log_writer,
                log_file_path,
                log_path,
                data_converter_config,
            )
            del log_writer
        gc.collect()

        if SORT_BY_TIMESTAMP:
            recording_table = open_arrow_table(log_file_path)
            recording_table = recording_table.sort_by([("timestamp", "ascending")])
            write_arrow_table(recording_table, log_file_path)

    return []


<<<<<<< HEAD
def get_av2_camera_metadata(log_path: Path) -> Dict[CameraType, PinholeCameraMetadata]:
=======
def get_av2_camera_metadata(log_path: Path) -> Dict[PinholeCameraType, PinholeCameraMetadata]:
>>>>>>> dev_v0.0.7

    intrinsics_file = log_path / "calibration" / "intrinsics.feather"
    intrinsics_df = pd.read_feather(intrinsics_file)

<<<<<<< HEAD
    camera_metadata: Dict[CameraType, PinholeCameraMetadata] = {}
=======
    camera_metadata: Dict[PinholeCameraType, PinholeCameraMetadata] = {}
>>>>>>> dev_v0.0.7
    for _, row in intrinsics_df.iterrows():
        row = row.to_dict()
        camera_type = AV2_CAMERA_TYPE_MAPPING[row["sensor_name"]]
        camera_metadata[camera_type] = PinholeCameraMetadata(
            camera_type=camera_type,
            width=row["width_px"],
            height=row["height_px"],
            intrinsics=PinholeIntrinsics(
                fx=row["fx_px"],
                fy=row["fy_px"],
                cx=row["cx_px"],
                cy=row["cy_px"],
            ),
            distortion=PinholeDistortion(
                k1=row["k1"],
                k2=row["k2"],
                p1=0.0,
                p2=0.0,
                k3=row["k3"],
            ),
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
    log_writer: ArrowLogWriter,
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

    for lidar_timestamp_ns in lidar_timestamps_ns:

        ego_state = _extract_av2_sensor_ego_state(city_se3_egovehicle_df, lidar_timestamp_ns)
        log_writer.add_row(
            token=create_token(str(lidar_timestamp_ns)),
            timestamp=TimePoint.from_ns(int(lidar_timestamp_ns)),
            ego_state=ego_state,
            box_detections=_extract_av2_sensor_box_detections(annotations_df, lidar_timestamp_ns, ego_state),
            traffic_lights=None,  # NOTE: Traffic light information is not available in AV2 sensor dataset.
            cameras=_extract_av2_sensor_camera(
                lidar_timestamp_ns,
                egovehicle_se3_sensor_df,
                synchronization_df,
                source_log_path,
                data_converter_config,
            ),
            lidars=None,
            scenario_tags=None,
            route_lane_group_ids=None,  # NOTE: Route information is not available in AV2 sensor dataset.
        )

    log_writer.close()


def _extract_av2_sensor_box_detections(
    annotations_df: Optional[pd.DataFrame],
    lidar_timestamp_ns: int,
    ego_state_se3: EgoStateSE3,
) -> BoxDetectionWrapper:

    # TODO: Extract velocity from annotations_df if available.

    if annotations_df is None:
        return BoxDetectionWrapper(box_detections=[])

    annotations_slice = get_slice_with_timestamp_ns(annotations_df, lidar_timestamp_ns)
    num_detections = len(annotations_slice)

    detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    detections_token: List[str] = annotations_slice["track_uuid"].tolist()
    detections_types: List[DetectionType] = []

    for detection_idx, (_, row) in enumerate(annotations_slice.iterrows()):
        row = row.to_dict()

        detections_state[detection_idx, BoundingBoxSE3Index.XYZ] = [row["tx_m"], row["ty_m"], row["tz_m"]]
        detections_state[detection_idx, BoundingBoxSE3Index.QUATERNION] = [row["qw"], row["qx"], row["qy"], row["qz"]]
        detections_state[detection_idx, BoundingBoxSE3Index.EXTENT] = [row["length_m"], row["width_m"], row["height_m"]]

        av2_detection_type = AV2SensorBoxDetectionType.deserialize(row["category"])
        detections_types.append(AV2_TO_DETECTION_TYPE[av2_detection_type])

    detections_state[:, BoundingBoxSE3Index.STATE_SE3] = convert_relative_to_absolute_se3_array(
        origin=ego_state_se3.rear_axle_se3,
        se3_array=detections_state[:, BoundingBoxSE3Index.STATE_SE3],
    )

    box_detections: List[BoxDetectionSE3] = []
    for detection_idx in range(num_detections):
        box_detections.append(
            BoxDetectionSE3(
                metadata=BoxDetectionMetadata(
                    detection_type=detections_types[detection_idx],
                    timepoint=None,
                    track_token=detections_token[detection_idx],
                    confidence=None,
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )

    return BoxDetectionWrapper(box_detections=box_detections)


def _extract_av2_sensor_ego_state(city_se3_egovehicle_df: pd.DataFrame, lidar_timestamp_ns: int) -> EgoStateSE3:
    ego_state_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, lidar_timestamp_ns)
    assert (
        len(ego_state_slice) == 1
    ), f"Expected exactly one ego state for timestamp {lidar_timestamp_ns}, got {len(ego_state_slice)}."

    ego_pose_dict = ego_state_slice.iloc[0].to_dict()
    rear_axle_pose = StateSE3(
        x=ego_pose_dict["tx_m"],
        y=ego_pose_dict["ty_m"],
        z=ego_pose_dict["tz_m"],
        qw=ego_pose_dict["qw"],
        qx=ego_pose_dict["qx"],
        qy=ego_pose_dict["qy"],
        qz=ego_pose_dict["qz"],
    )

    vehicle_parameters = get_av2_ford_fusion_hybrid_parameters()
    center = rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_pose, vehicle_parameters=vehicle_parameters)

    # TODO: Add script to calculate the dynamic state from log sequence.
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
    )

    return EgoStateSE3(
        center_se3=center,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,
    )


def _extract_av2_sensor_camera(
    lidar_timestamp_ns: int,
    egovehicle_se3_sensor_df: pd.DataFrame,
    synchronization_df: pd.DataFrame,
    source_log_path: Path,
    data_converter_config: DataConverterConfig,
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]]:

    camera_dict: Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]] = {}
    split = source_log_path.parent.name
    log_id = source_log_path.name

    source_dataset_dir = source_log_path.parent.parent

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
        if relative_image_path is not None:
            absolute_image_path = source_dataset_dir / relative_image_path
            assert absolute_image_path.exists()

            # TODO: Adjust for finer IMU timestamps to correct the camera extrinsic.
            camera_extrinsic = StateSE3(
                x=row["tx_m"],
                y=row["ty_m"],
                z=row["tz_m"],
                qw=row["qw"],
                qx=row["qx"],
                qy=row["qy"],
                qz=row["qz"],
            )
            camera_data = None
            if data_converter_config.camera_store_option == "path":
                camera_data = str(relative_image_path)
            elif data_converter_config.camera_store_option == "binary":
                with open(absolute_image_path, "rb") as f:
                    camera_data = f.read()
            camera_dict[camera_type] = camera_data, camera_extrinsic

    return camera_dict


def _extract_lidar(lidar_pc, data_converter_config: DataConverterConfig) -> Dict[LiDARType, Optional[str]]:
    # TODO: Implement this function to extract lidar data.
    return {}
