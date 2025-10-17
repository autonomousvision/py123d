from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.av2.utils.av2_constants import (
    AV2_CAMERA_TYPE_MAPPING,
    AV2_SENSOR_SPLITS,
    AV2_TO_DETECTION_TYPE,
    AV2SensorBoxDetectionType,
)
from py123d.conversion.datasets.av2.utils.av2_helper import (
    build_sensor_dataframe,
    build_synchronization_dataframe,
    find_closest_target_fpath,
    get_slice_with_timestamp_ns,
)
from py123d.conversion.datasets.av2.utils.av2_map_conversion import convert_av2_map
from py123d.conversion.datasets.av2.utils.av2_sensor_loading import load_av2_sensor_lidar_pc_from_path
from py123d.conversion.log_writer.abstract_log_writer import AbstractLogWriter
from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.utils.sensor_utils.lidar_index_registry import AVSensorLidarIndex
from py123d.datatypes.detections.detection import BoxDetectionMetadata, BoxDetectionSE3, BoxDetectionWrapper
from py123d.datatypes.detections.detection_types import DetectionType
from py123d.datatypes.maps.map_metadata import MapMetadata
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from py123d.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import (
    get_av2_ford_fusion_hybrid_parameters,
    rear_axle_se3_to_center_se3,
)
from py123d.geometry import BoundingBoxSE3Index, StateSE3, Vector3D, Vector3DIndex
from py123d.geometry.bounding_box import BoundingBoxSE3
from py123d.geometry.transform.transform_se3 import convert_relative_to_absolute_se3_array


class AV2SensorConverter(AbstractDatasetConverter):
    def __init__(
        self,
        splits: List[str],
        av2_data_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        super().__init__(dataset_converter_config)
        for split in splits:
            assert (
                split in AV2_SENSOR_SPLITS
            ), f"Split {split} is not available. Available splits: {self.available_splits}"

        self._splits: List[str] = splits
        self._av2_data_root: Path = Path(av2_data_root)
        self._log_paths_and_split: Dict[str, List[Path]] = self._collect_log_paths()

    def _collect_log_paths(self) -> Dict[str, List[Path]]:
        log_paths_and_split: List[Tuple[Path, str]] = []

        for split in self._splits:
            dataset_name = split.split("_")[0]
            split_type = split.split("_")[-1]
            assert split_type in ["train", "val", "test"]

            if "av2-sensor" == dataset_name:
                log_folder = self._av2_data_root / "sensor" / split_type
            else:
                raise ValueError(f"Unknown dataset name {dataset_name} in split {split}.")

            log_paths_and_split.extend([(log_path, split) for log_path in log_folder.iterdir()])

        return log_paths_and_split

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(self._log_paths_and_split)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._log_paths_and_split)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""

        source_log_path, split = self._log_paths_and_split[map_index]

        # 1. Initialize map metadata
        map_metadata = _get_av2_sensor_map_metadata(split, source_log_path)

        # 2. Prepare map writer
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)

        # 3. Process source map data
        if map_needs_writing:
            convert_av2_map(source_log_path, map_writer)

        # 4. Finalize map writing
        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        source_log_path, split = self._log_paths_and_split[log_index]

        # 1. Initialize Metadata
        map_metadata = _get_av2_sensor_map_metadata(split, source_log_path)
        log_metadata = LogMetadata(
            dataset="av2-sensor",
            split=split,
            log_name=source_log_path.name,
            location=map_metadata.location,
            timestep_seconds=0.1,
            vehicle_parameters=get_av2_ford_fusion_hybrid_parameters(),
            camera_metadata=_get_av2_camera_metadata(source_log_path, self.dataset_converter_config),
            lidar_metadata=_get_av2_lidar_metadata(source_log_path, self.dataset_converter_config),
            map_metadata=map_metadata,
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        # 3. Process source log data
        if log_needs_writing:

            sensor_df = build_sensor_dataframe(source_log_path)
            synchronization_df = build_synchronization_dataframe(sensor_df)

            lidar_sensor = sensor_df.xs(key="lidar", level=2)
            lidar_timestamps_ns = np.sort([int(idx_tuple[2]) for idx_tuple in lidar_sensor.index])

            annotations_df = (
                pd.read_feather(source_log_path / "annotations.feather")
                if (source_log_path / "annotations.feather").exists()
                else None
            )
            city_se3_egovehicle_df = pd.read_feather(source_log_path / "city_SE3_egovehicle.feather")
            egovehicle_se3_sensor_df = pd.read_feather(
                source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
            )

            for lidar_timestamp_ns in lidar_timestamps_ns:
                ego_state = _extract_av2_sensor_ego_state(city_se3_egovehicle_df, lidar_timestamp_ns)
                log_writer.write(
                    timestamp=TimePoint.from_ns(int(lidar_timestamp_ns)),
                    ego_state=ego_state,
                    box_detections=_extract_av2_sensor_box_detections(annotations_df, lidar_timestamp_ns, ego_state),
                    cameras=_extract_av2_sensor_camera(
                        lidar_timestamp_ns,
                        egovehicle_se3_sensor_df,
                        synchronization_df,
                        source_log_path,
                        self.dataset_converter_config,
                    ),
                    lidars=_extract_av2_sensor_lidars(
                        source_log_path,
                        lidar_timestamp_ns,
                        self.dataset_converter_config,
                    ),
                )

        # 4. Finalize log writing
        log_writer.close()


def _get_av2_sensor_map_metadata(split: str, source_log_path: Path) -> MapMetadata:
    # NOTE: We need to get the city name from the map folder.
    # see: https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/sensor/av2_sensor_dataloader.py#L163
    map_folder = source_log_path / "map"
    log_map_archive_path = next(map_folder.glob("log_map_archive_*.json"))
    location = log_map_archive_path.name.split("____")[1].split("_")[0]
    return MapMetadata(
        dataset="av2-sensor",
        split=split,
        log_name=source_log_path.name,
        location=location,  # TODO: Add location information, e.g. city name.
        map_has_z=True,
        map_is_local=True,
    )


def _get_av2_camera_metadata(
    source_log_path: Path, dataset_converter_config: DatasetConverterConfig
) -> Dict[PinholeCameraType, PinholeCameraMetadata]:

    camera_metadata: Dict[PinholeCameraType, PinholeCameraMetadata] = {}

    if dataset_converter_config.include_cameras:
        intrinsics_file = source_log_path / "calibration" / "intrinsics.feather"
        intrinsics_df = pd.read_feather(intrinsics_file)
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


def _get_av2_lidar_metadata(
    source_log_path: Path, dataset_converter_config: DatasetConverterConfig
) -> Dict[LiDARType, LiDARMetadata]:

    metadata: Dict[LiDARType, LiDARMetadata] = {}

    if dataset_converter_config.include_lidars:

        # Load calibration feather file
        calibration_file = source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
        calibration_df = pd.read_feather(calibration_file)

        # NOTE: AV2 has two two stacked lidars: up_lidar and down_lidar.
        # We store these as separate LiDARType entries.

        # top lidar:
        metadata[LiDARType.LIDAR_TOP] = LiDARMetadata(
            lidar_type=LiDARType.LIDAR_TOP,
            lidar_index=AVSensorLidarIndex,
            extrinsic=_row_dict_to_state_se3(
                calibration_df[calibration_df["sensor_name"] == "up_lidar"].iloc[0].to_dict()
            ),
        )
        # down lidar:
        metadata[LiDARType.LIDAR_DOWN] = LiDARMetadata(
            lidar_type=LiDARType.LIDAR_DOWN,
            lidar_index=AVSensorLidarIndex,
            extrinsic=_row_dict_to_state_se3(
                calibration_df[calibration_df["sensor_name"] == "down_lidar"].iloc[0].to_dict()
            ),
        )
    return metadata


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
    rear_axle_pose = _row_dict_to_state_se3(ego_pose_dict)

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
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]]:

    camera_dict: Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]] = {}
    split = source_log_path.parent.name
    log_id = source_log_path.name

    if dataset_converter_config.include_cameras:
        av2_sensor_data_root = source_log_path.parent.parent

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
                absolute_image_path = av2_sensor_data_root / relative_image_path
                assert absolute_image_path.exists()

                # TODO: Adjust for finer IMU timestamps to correct the camera extrinsic.
                camera_extrinsic = _row_dict_to_state_se3(row)
                camera_data = None
                if dataset_converter_config.camera_store_option == "path":
                    camera_data = str(relative_image_path)
                elif dataset_converter_config.camera_store_option == "binary":
                    with open(absolute_image_path, "rb") as f:
                        camera_data = f.read()
                camera_dict[camera_type] = camera_data, camera_extrinsic

    return camera_dict


def _extract_av2_sensor_lidars(
    source_log_path: Path, lidar_timestamp_ns: int, dataset_converter_config: DatasetConverterConfig
) -> Optional[Dict[LiDARType, Union[str, npt.NDArray[np.float32]]]]:
    lidar_dict: Dict[LiDARType, Union[str, npt.NDArray[np.float32]]] = {}
    if dataset_converter_config.include_lidars:
        av2_sensor_data_root = source_log_path.parent.parent
        split_type = source_log_path.parent.name
        log_name = source_log_path.name

        relative_feather_path = f"{split_type}/{log_name}/sensors/lidar/{lidar_timestamp_ns}.feather"
        lidar_feather_path = av2_sensor_data_root / relative_feather_path
        # if lidar_feather_path.exists():

        assert lidar_feather_path.exists(), f"LiDAR feather file not found: {lidar_feather_path}"
        if dataset_converter_config.lidar_store_option == "path":
            # NOTE: It is somewhat inefficient to store the same path for both lidars,
            # but this keeps the interface simple for now.
            lidar_dict = {
                LiDARType.LIDAR_TOP: str(relative_feather_path),
                LiDARType.LIDAR_DOWN: str(relative_feather_path),
            }
        elif dataset_converter_config.lidar_store_option == "binary":
            # NOTE: For binary storage, we pass the point cloud to the log writer.
            # Compression is handled internally in the log writer.
            lidar_dict: Dict[LiDARType, np.ndarray] = load_av2_sensor_lidar_pc_from_path(lidar_feather_path)
    return lidar_dict


def _row_dict_to_state_se3(row_dict: Dict[str, float]) -> StateSE3:
    """Helper function to convert a row dictionary to a StateSE3 object."""
    return StateSE3(
        x=row_dict["tx_m"],
        y=row_dict["ty_m"],
        z=row_dict["tz_m"],
        qw=row_dict["qw"],
        qx=row_dict["qx"],
        qy=row_dict["qy"],
        qz=row_dict["qz"],
    )
