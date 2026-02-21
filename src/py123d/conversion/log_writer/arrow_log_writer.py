import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.utils.arrow_metadata_utils import add_log_metadata_to_arrow_schema
from py123d.common.dataset_paths import get_dataset_paths
from py123d.common.utils.arrow_column_names import (
    BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN,
    BOX_DETECTIONS_LABEL_COLUMN,
    BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN,
    BOX_DETECTIONS_TOKEN_COLUMN,
    BOX_DETECTIONS_VELOCITY_3D_COLUMN,
    EGO_DYNAMIC_STATE_SE3_COLUMN,
    EGO_REAR_AXLE_SE3_COLUMN,
    FISHEYE_CAMERA_DATA_COLUMN,
    FISHEYE_CAMERA_EXTRINSIC_COLUMN,
    FISHEYE_CAMERA_TIMESTAMP_COLUMN,
    LIDAR_DATA_COLUMN,
    PINHOLE_CAMERA_DATA_COLUMN,
    PINHOLE_CAMERA_EXTRINSIC_COLUMN,
    PINHOLE_CAMERA_TIMESTAMP_COLUMN,
    ROUTE_LANE_GROUP_IDS_COLUMN,
    SCENARIO_TAGS_COLUMN,
    TIMESTAMP_US_COLUMN,
    TRAFFIC_LIGHTS_LANE_ID_COLUMN,
    TRAFFIC_LIGHTS_STATUS_COLUMN,
    UUID_COLUMN,
)
from py123d.common.utils.arrow_file_names import (
    BOX_DETECTIONS_FILE,
    EGO_STATE_FILE,
    FISHEYE_CAMERA_FILE,
    INDEX_FILE,
    LIDAR_FILE,
    PINHOLE_CAMERA_FILE,
    ROUTE_FILE,
    SCENARIO_TAGS_FILE,
    TRAFFIC_LIGHTS_FILE,
)
from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.conversion.abstract_dataset_converter import AbstractLogWriter, DatasetConverterConfig
from py123d.conversion.log_writer.abstract_log_writer import CameraData, LiDARData
from py123d.conversion.sensor_io.camera.jpeg_camera_io import (
    decode_image_from_jpeg_binary,
    encode_image_as_jpeg_binary,
    load_image_from_jpeg_file,
    load_jpeg_binary_from_jpeg_file,
)
from py123d.conversion.sensor_io.camera.mp4_camera_io import MP4Writer
from py123d.conversion.sensor_io.camera.png_camera_io import (
    encode_image_as_png_binary,
    load_image_from_png_file,
    load_png_binary_from_png_file,
)
from py123d.conversion.sensor_io.lidar.draco_lidar_io import encode_lidar_pc_as_draco_binary
from py123d.conversion.sensor_io.lidar.file_lidar_io import load_lidar_pcs_from_file
from py123d.conversion.sensor_io.lidar.laz_lidar_io import encode_lidar_pc_as_laz_binary
from py123d.datatypes.detections.box_detections import BoxDetectionSE3, BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.metadata import LogMetadata
from py123d.datatypes.sensors import LiDARType, PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3Index
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import BoundingBoxSE3Index, PoseSE3, PoseSE3Index, Vector3DIndex


def _get_logs_root() -> Path:
    logs_root = get_dataset_paths().py123d_logs_root
    assert logs_root is not None, "PY123D_DATA_ROOT must be set."
    return logs_root


def _get_sensors_root() -> Path:
    sensors_root = get_dataset_paths().py123d_sensors_root
    assert sensors_root is not None, "PY123D_DATA_ROOT must be set."
    return sensors_root


def _store_option_to_arrow_type(
    store_option: Literal["path", "jpeg_binary", "png_binary", "laz_binary", "draco_binary", "mp4"],
) -> pa.DataType:
    """Maps the store option literal to the corresponding Arrow data type."""
    data_type_map = {
        "path": pa.string(),
        "jpeg_binary": pa.binary(),
        "png_binary": pa.binary(),
        "laz_binary": pa.binary(),
        "draco_binary": pa.binary(),
        "mp4": pa.int64(),
    }
    return data_type_map[store_option]


def _get_uuid_arrow_type():
    """Gets the appropriate Arrow UUID data type based on pyarrow version."""
    # NOTE @DanielDauner: pyarrow introduced native UUID type in version 18.0.0
    # Easiest option is to require this version or higher, but thanks to the Waymo dataset that's not possible. :(
    if pa.__version__ >= "18.0.0":
        return pa.uuid()
    else:
        return pa.binary(16)


# ------------------------------------------------------------------------------------------------------------------
# Per-modality schema builders
# ------------------------------------------------------------------------------------------------------------------


def _build_index_schema(log_metadata: LogMetadata) -> pa.Schema:
    """Schema for index.arrow: uuid + timestamp_us. Carries LogMetadata in schema metadata."""
    schema = pa.schema(
        [
            (UUID_COLUMN, _get_uuid_arrow_type()),
            (TIMESTAMP_US_COLUMN, pa.int64()),
        ]
    )
    return add_log_metadata_to_arrow_schema(schema, log_metadata)


def _build_ego_state_schema() -> pa.Schema:
    """Schema for EgoState.arrow."""
    return pa.schema(
        [
            (EGO_REAR_AXLE_SE3_COLUMN, pa.list_(pa.float64(), len(PoseSE3Index))),
            (EGO_DYNAMIC_STATE_SE3_COLUMN, pa.list_(pa.float64(), len(DynamicStateSE3Index))),
        ]
    )


def _build_box_detections_schema() -> pa.Schema:
    """Schema for BoxDetections.arrow."""
    return pa.schema(
        [
            (BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN, pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
            (BOX_DETECTIONS_TOKEN_COLUMN, pa.list_(pa.string())),
            (BOX_DETECTIONS_LABEL_COLUMN, pa.list_(pa.int16())),
            (BOX_DETECTIONS_VELOCITY_3D_COLUMN, pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
            (BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN, pa.list_(pa.int64())),
        ]
    )


def _build_traffic_lights_schema() -> pa.Schema:
    """Schema for TrafficLights.arrow."""
    return pa.schema(
        [
            (TRAFFIC_LIGHTS_LANE_ID_COLUMN, pa.list_(pa.int64())),
            (TRAFFIC_LIGHTS_STATUS_COLUMN, pa.list_(pa.int16())),
        ]
    )


def _build_pinhole_camera_schema(camera_name: str, store_option: str) -> pa.Schema:
    """Schema for a single PinholeCamera.{name}.arrow file."""
    return pa.schema(
        [
            (PINHOLE_CAMERA_DATA_COLUMN(camera_name), _store_option_to_arrow_type(store_option)),
            (PINHOLE_CAMERA_EXTRINSIC_COLUMN(camera_name), pa.list_(pa.float64(), len(PoseSE3Index))),
            (PINHOLE_CAMERA_TIMESTAMP_COLUMN(camera_name), pa.int64()),
        ]
    )


def _build_fisheye_camera_schema(camera_name: str, store_option: str) -> pa.Schema:
    """Schema for a single FisheyeCamera.{name}.arrow file."""
    return pa.schema(
        [
            (FISHEYE_CAMERA_DATA_COLUMN(camera_name), _store_option_to_arrow_type(store_option)),
            (FISHEYE_CAMERA_EXTRINSIC_COLUMN(camera_name), pa.list_(pa.float64(), len(PoseSE3Index))),
            (FISHEYE_CAMERA_TIMESTAMP_COLUMN(camera_name), pa.int64()),
        ]
    )


def _build_lidar_schema(lidar_name: str, store_option: str) -> pa.Schema:
    """Schema for a single LiDAR.{name}.arrow file."""
    return pa.schema(
        [
            (LIDAR_DATA_COLUMN(lidar_name), _store_option_to_arrow_type(store_option)),
        ]
    )


def _build_scenario_tags_schema() -> pa.Schema:
    """Schema for ScenarioTags.arrow."""
    return pa.schema([(SCENARIO_TAGS_COLUMN, pa.list_(pa.string()))])


def _build_route_schema() -> pa.Schema:
    """Schema for Route.arrow."""
    return pa.schema([(ROUTE_LANE_GROUP_IDS_COLUMN, pa.list_(pa.int64()))])


# ------------------------------------------------------------------------------------------------------------------
# Internal modality writer
# ------------------------------------------------------------------------------------------------------------------


class _ModalityWriter:
    """Manages a single Arrow IPC file for one modality."""

    def __init__(self, file_path: Path, schema: pa.Schema, compression: Optional[pa.Codec] = None) -> None:
        self._file_path = file_path
        self._schema = schema
        self._source = pa.OSFile(str(file_path), "wb")
        options = pa.ipc.IpcWriteOptions(compression=compression)
        self._writer = pa.ipc.new_file(self._source, schema=schema, options=options)

    def write_batch(self, data: Dict[str, Any]) -> None:
        """Write a single-row record batch from a dict."""
        batch = pa.record_batch(data, schema=self._schema)
        self._writer.write_batch(batch)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._source is not None:
            self._source.close()
            self._source = None


# ------------------------------------------------------------------------------------------------------------------
# ArrowLogWriter (modular folder-per-log)
# ------------------------------------------------------------------------------------------------------------------


class ArrowLogWriter(AbstractLogWriter):
    """Log writer for Arrow-based logs. Writes each modality to a separate Arrow IPC file within a log directory."""

    def __init__(
        self,
        logs_root: Optional[Union[str, Path]] = None,
        sensors_root: Optional[Union[str, Path]] = None,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        """Initializes the :class:`ArrowLogWriter`.

        :param logs_root: The root directory for logs, defaults to None
        :param sensors_root: The root directory for sensors (i.e. in case of re-writing sensor files), defaults to None
        :param ipc_compression: The IPC compression method, defaults to None
        :param ipc_compression_level: The IPC compression level, defaults to None
        """

        self._logs_root = Path(logs_root) if logs_root is not None else _get_logs_root()
        self._sensors_root = Path(sensors_root) if sensors_root is not None else _get_sensors_root()
        self._ipc_compression = ipc_compression
        self._ipc_compression_level = ipc_compression_level

        # Loaded during .reset() and cleared during .close()
        self._dataset_converter_config: Optional[DatasetConverterConfig] = None
        self._log_metadata: Optional[LogMetadata] = None
        self._log_dir: Optional[Path] = None
        self._modality_writers: Dict[str, _ModalityWriter] = {}
        self._pinhole_mp4_writers: Dict[str, MP4Writer] = {}
        self._fisheye_mei_mp4_writers: Dict[str, MP4Writer] = {}

    def _get_compression(self) -> Optional[pa.Codec]:
        """Returns the IPC compression codec, or None if no compression is configured."""
        if self._ipc_compression is not None:
            return pa.Codec(self._ipc_compression, compression_level=self._ipc_compression_level)
        return None

    def reset(self, dataset_converter_config: DatasetConverterConfig, log_metadata: LogMetadata) -> bool:
        """Inherited, see superclass."""

        log_needs_writing: bool = False
        log_dir: Path = self._logs_root / log_metadata.split / log_metadata.log_name

        # Check if the log directory already exists or needs to be overwritten
        index_path = log_dir / INDEX_FILE
        if not index_path.exists() or dataset_converter_config.force_log_conversion:
            log_needs_writing = True

            # Delete the directory if it exists (clean start)
            if log_dir.exists():
                shutil.rmtree(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Store config and metadata
            self._dataset_converter_config = dataset_converter_config
            self._log_metadata = log_metadata
            self._log_dir = log_dir

            compression = self._get_compression()

            # Always create the index writer (uuid + timestamp + metadata in schema)
            self._modality_writers["index"] = _ModalityWriter(
                log_dir / INDEX_FILE, _build_index_schema(log_metadata), compression
            )

            # Conditionally create modality writers based on config
            if dataset_converter_config.include_ego:
                self._modality_writers["ego_state"] = _ModalityWriter(
                    log_dir / EGO_STATE_FILE, _build_ego_state_schema(), compression
                )

            if dataset_converter_config.include_box_detections:
                self._modality_writers["box_detections"] = _ModalityWriter(
                    log_dir / BOX_DETECTIONS_FILE, _build_box_detections_schema(), compression
                )

            if dataset_converter_config.include_traffic_lights:
                self._modality_writers["traffic_lights"] = _ModalityWriter(
                    log_dir / TRAFFIC_LIGHTS_FILE, _build_traffic_lights_schema(), compression
                )

            if dataset_converter_config.include_pinhole_cameras:
                for pinhole_camera_type in log_metadata.pinhole_camera_metadata.keys():
                    camera_name = pinhole_camera_type.serialize()
                    writer_key = f"pinhole_{camera_name}"
                    self._modality_writers[writer_key] = _ModalityWriter(
                        log_dir / PINHOLE_CAMERA_FILE(camera_name),
                        _build_pinhole_camera_schema(camera_name, dataset_converter_config.pinhole_camera_store_option),
                        compression,
                    )

            if dataset_converter_config.include_fisheye_mei_cameras:
                for fisheye_camera_type in log_metadata.fisheye_mei_camera_metadata.keys():
                    camera_name = fisheye_camera_type.serialize()
                    writer_key = f"fisheye_{camera_name}"
                    self._modality_writers[writer_key] = _ModalityWriter(
                        log_dir / FISHEYE_CAMERA_FILE(camera_name),
                        _build_fisheye_camera_schema(
                            camera_name, dataset_converter_config.fisheye_mei_camera_store_option
                        ),
                        compression,
                    )

            if dataset_converter_config.include_lidars and len(log_metadata.lidar_metadata) > 0:
                if dataset_converter_config.lidar_store_option == "path_merged":
                    lidar_name = LiDARType.LIDAR_MERGED.serialize()
                    self._modality_writers["lidar_LIDAR_MERGED"] = _ModalityWriter(
                        log_dir / LIDAR_FILE(lidar_name),
                        _build_lidar_schema(lidar_name, "path"),
                        compression,
                    )
                else:
                    for lidar_type in log_metadata.lidar_metadata.keys():
                        lidar_name = lidar_type.serialize()
                        writer_key = f"lidar_{lidar_name}"
                        self._modality_writers[writer_key] = _ModalityWriter(
                            log_dir / LIDAR_FILE(lidar_name),
                            _build_lidar_schema(lidar_name, dataset_converter_config.lidar_store_option),
                            compression,
                        )

            if dataset_converter_config.include_scenario_tags:
                self._modality_writers["scenario_tags"] = _ModalityWriter(
                    log_dir / SCENARIO_TAGS_FILE, _build_scenario_tags_schema(), compression
                )

            if dataset_converter_config.include_route:
                self._modality_writers["route"] = _ModalityWriter(
                    log_dir / ROUTE_FILE, _build_route_schema(), compression
                )

            self._pinhole_mp4_writers = {}
            self._fisheye_mei_mp4_writers = {}

        return log_needs_writing

    def write(
        self,
        timestamp: TimePoint,
        ego_state: Optional[EgoStateSE3] = None,
        box_detections: Optional[BoxDetectionWrapper] = None,
        traffic_lights: Optional[TrafficLightDetectionWrapper] = None,
        pinhole_cameras: Optional[List[CameraData]] = None,
        fisheye_mei_cameras: Optional[List[CameraData]] = None,
        lidars: Optional[List[LiDARData]] = None,
        scenario_tags: Optional[List[str]] = None,
        route_lane_group_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Inherited, see superclass."""

        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        assert "index" in self._modality_writers, "Log writer is not initialized."

        # ----------------------------------------------------------------------------------------------------------
        # Index (always written)
        # ----------------------------------------------------------------------------------------------------------
        self._modality_writers["index"].write_batch(
            {
                UUID_COLUMN: [
                    create_deterministic_uuid(
                        split=self._log_metadata.split,
                        log_name=self._log_metadata.log_name,
                        timestamp_us=timestamp.time_us,
                    ).bytes
                ],
                TIMESTAMP_US_COLUMN: [timestamp.time_us],
            }
        )

        # ----------------------------------------------------------------------------------------------------------
        # Ego State
        # ----------------------------------------------------------------------------------------------------------
        if "ego_state" in self._modality_writers:
            assert ego_state is not None, "Ego state is required but not provided."
            self._modality_writers["ego_state"].write_batch(
                {
                    EGO_REAR_AXLE_SE3_COLUMN: [ego_state.rear_axle_se3],
                    EGO_DYNAMIC_STATE_SE3_COLUMN: [ego_state.dynamic_state_se3],
                }
            )

        # ----------------------------------------------------------------------------------------------------------
        # Box Detections
        # ----------------------------------------------------------------------------------------------------------
        if "box_detections" in self._modality_writers:
            assert box_detections is not None, "Box detections are required but not provided."

            box_detection_state = []
            box_detection_token = []
            box_detection_label = []
            box_detection_velocity = []
            box_detection_num_lidar_points = []

            for box_detection in box_detections:
                assert isinstance(box_detection, BoxDetectionSE3), "Currently only BoxDetectionSE3 is supported."
                box_detection_state.append(box_detection.bounding_box_se3)
                box_detection_token.append(box_detection.metadata.track_token)
                box_detection_label.append(int(box_detection.metadata.label))
                box_detection_velocity.append(box_detection.velocity_3d)
                box_detection_num_lidar_points.append(box_detection.metadata.num_lidar_points)

            self._modality_writers["box_detections"].write_batch(
                {
                    BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN: [box_detection_state],
                    BOX_DETECTIONS_TOKEN_COLUMN: [box_detection_token],
                    BOX_DETECTIONS_LABEL_COLUMN: [box_detection_label],
                    BOX_DETECTIONS_VELOCITY_3D_COLUMN: [box_detection_velocity],
                    BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN: [box_detection_num_lidar_points],
                }
            )

        # ----------------------------------------------------------------------------------------------------------
        # Traffic Lights
        # ----------------------------------------------------------------------------------------------------------
        if "traffic_lights" in self._modality_writers:
            assert traffic_lights is not None, "Traffic light detections are required but not provided."

            traffic_light_ids = []
            traffic_light_statuses = []
            for traffic_light in traffic_lights:
                traffic_light_ids.append(traffic_light.lane_id)
                traffic_light_statuses.append(int(traffic_light.status))

            self._modality_writers["traffic_lights"].write_batch(
                {
                    TRAFFIC_LIGHTS_LANE_ID_COLUMN: [traffic_light_ids],
                    TRAFFIC_LIGHTS_STATUS_COLUMN: [traffic_light_statuses],
                }
            )

        # ----------------------------------------------------------------------------------------------------------
        # Pinhole Cameras
        # ----------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_pinhole_cameras:
            assert pinhole_cameras is not None, "Pinhole camera data is required but not provided."
            provided_pinhole_data = self._prepare_camera_data_dict(
                pinhole_cameras, self._dataset_converter_config.pinhole_camera_store_option
            )
            provided_pinhole_extrinsics = {
                camera_data.camera_type: camera_data.extrinsic for camera_data in pinhole_cameras
            }
            provided_pinhole_timestamps = {
                camera_data.camera_type: camera_data.timestamp for camera_data in pinhole_cameras
            }
            expected_pinhole_cameras = set(self._log_metadata.pinhole_camera_metadata.keys())

            for pinhole_camera_type in expected_pinhole_cameras:
                pinhole_camera_name = pinhole_camera_type.serialize()
                writer_key = f"pinhole_{pinhole_camera_name}"

                # NOTE @DanielDauner: Missing cameras are allowed, e.g., for synchronization mismatches.
                # In this case, we write None/null to the arrow table.
                pinhole_camera_data: Optional[Any] = None
                pinhole_camera_pose: Optional[PoseSE3] = None
                pinhole_camera_timestamp: Optional[TimePoint] = None
                if pinhole_camera_type in provided_pinhole_data:
                    pinhole_camera_data = provided_pinhole_data[pinhole_camera_type]
                    pinhole_camera_pose = provided_pinhole_extrinsics[pinhole_camera_type]
                    pinhole_camera_timestamp = provided_pinhole_timestamps[pinhole_camera_type]

                self._modality_writers[writer_key].write_batch(
                    {
                        PINHOLE_CAMERA_DATA_COLUMN(pinhole_camera_name): [pinhole_camera_data],
                        PINHOLE_CAMERA_EXTRINSIC_COLUMN(pinhole_camera_name): [
                            pinhole_camera_pose.array if pinhole_camera_pose is not None else None
                        ],
                        PINHOLE_CAMERA_TIMESTAMP_COLUMN(pinhole_camera_name): [
                            pinhole_camera_timestamp.time_us if pinhole_camera_timestamp is not None else None
                        ],
                    }
                )

        # ----------------------------------------------------------------------------------------------------------
        # Fisheye MEI Cameras
        # ----------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_fisheye_mei_cameras:
            assert fisheye_mei_cameras is not None, "Fisheye MEI camera data is required but not provided."
            provided_fisheye_mei_data = self._prepare_camera_data_dict(
                fisheye_mei_cameras, self._dataset_converter_config.fisheye_mei_camera_store_option
            )
            provided_fisheye_mei_extrinsics = {
                camera_data.camera_type: camera_data.extrinsic for camera_data in fisheye_mei_cameras
            }
            provided_fisheye_mei_timestamps = {
                camera_data.camera_type: camera_data.timestamp for camera_data in fisheye_mei_cameras
            }
            expected_fisheye_mei_cameras = set(self._log_metadata.fisheye_mei_camera_metadata.keys())

            for fisheye_mei_camera_type in expected_fisheye_mei_cameras:
                fisheye_mei_camera_name = fisheye_mei_camera_type.serialize()
                writer_key = f"fisheye_{fisheye_mei_camera_name}"

                # NOTE @DanielDauner: Missing cameras are allowed, e.g., for synchronization mismatches.
                # In this case, we write None/null to the arrow table.
                fisheye_mei_camera_data: Optional[Any] = None
                fisheye_mei_camera_pose: Optional[PoseSE3] = None
                fisheye_mei_camera_timestamp: Optional[TimePoint] = None
                if fisheye_mei_camera_type in provided_fisheye_mei_data:
                    fisheye_mei_camera_data = provided_fisheye_mei_data[fisheye_mei_camera_type]
                    fisheye_mei_camera_pose = provided_fisheye_mei_extrinsics[fisheye_mei_camera_type]
                    fisheye_mei_camera_timestamp = provided_fisheye_mei_timestamps[fisheye_mei_camera_type]

                self._modality_writers[writer_key].write_batch(
                    {
                        FISHEYE_CAMERA_DATA_COLUMN(fisheye_mei_camera_name): [fisheye_mei_camera_data],
                        FISHEYE_CAMERA_EXTRINSIC_COLUMN(fisheye_mei_camera_name): [
                            fisheye_mei_camera_pose.array if fisheye_mei_camera_pose is not None else None
                        ],
                        FISHEYE_CAMERA_TIMESTAMP_COLUMN(fisheye_mei_camera_name): [
                            fisheye_mei_camera_timestamp.time_us if fisheye_mei_camera_timestamp is not None else None
                        ],
                    }
                )

        # ----------------------------------------------------------------------------------------------------------
        # LiDARs
        # ----------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_lidars and len(self._log_metadata.lidar_metadata) > 0:
            assert lidars is not None, "LiDAR data is required but not provided."

            if self._dataset_converter_config.lidar_store_option == "path_merged":
                # NOTE @DanielDauner: The path_merged option is necessary for datasets, that natively store multiple
                # LiDAR point clouds in a single file. In this case, writing the file path several times is wasteful.
                # Instead, we store the file path once, and divide the point clouds during reading.
                assert len(lidars) <= 1, "Exactly one LiDAR data must be provided for merged LiDAR storage."

                if len(lidars) == 0:
                    merged_lidar_data: Optional[str] = None
                else:
                    assert lidars[0].has_file_path, "LiDAR data must provide file path for merged LiDAR storage."
                    merged_lidar_data: Optional[str] = str(lidars[0].relative_path)
                lidar_name = LiDARType.LIDAR_MERGED.serialize()
                self._modality_writers["lidar_LIDAR_MERGED"].write_batch(
                    {LIDAR_DATA_COLUMN(lidar_name): [merged_lidar_data]}
                )

            else:
                expected_lidars = set(self._log_metadata.lidar_metadata.keys())
                lidar_data_dict = self._prepare_lidar_data_dict(lidars)

                for lidar_type in expected_lidars:
                    lidar_name = lidar_type.serialize()
                    writer_key = f"lidar_{lidar_name}"
                    lidar_data: Optional[Union[str, bytes]] = lidar_data_dict.get(lidar_type, None)
                    self._modality_writers[writer_key].write_batch({LIDAR_DATA_COLUMN(lidar_name): [lidar_data]})

        # ----------------------------------------------------------------------------------------------------------
        # Miscellaneous (Scenario Tags / Route)
        # ----------------------------------------------------------------------------------------------------------
        if "scenario_tags" in self._modality_writers:
            assert scenario_tags is not None, "Scenario tags are required but not provided."
            self._modality_writers["scenario_tags"].write_batch({SCENARIO_TAGS_COLUMN: [scenario_tags]})

        if "route" in self._modality_writers:
            assert route_lane_group_ids is not None, "Route lane group IDs are required but not provided."
            self._modality_writers["route"].write_batch({ROUTE_LANE_GROUP_IDS_COLUMN: [route_lane_group_ids]})

    def close(self) -> None:
        """Inherited, see superclass."""
        for writer in self._modality_writers.values():
            writer.close()
        self._modality_writers = {}

        self._dataset_converter_config = None
        self._log_metadata = None
        self._log_dir = None

        for mp4_writer in self._pinhole_mp4_writers.values():
            mp4_writer.close()
        self._pinhole_mp4_writers = {}
        for mp4_writer in self._fisheye_mei_mp4_writers.values():
            mp4_writer.close()
        self._fisheye_mei_mp4_writers = {}

    def _prepare_lidar_data_dict(self, lidars: List[LiDARData]) -> Dict[LiDARType, Union[str, bytes]]:
        """Helper function to prepare LiDAR data dictionary for the target storage option.

        :param lidars: List of LiDARData objects to be processed.
        :return: Dictionary mapping LiDARType to either file path (str) or binary data (bytes) depending on storage option.
        """

        lidar_data_dict: Dict[LiDARType, Union[str, bytes]] = {}

        if self._dataset_converter_config.lidar_store_option == "path":
            for lidar_data in lidars:
                assert lidar_data.has_file_path, "LiDAR data must provide file path for path storage."
                lidar_data_dict[lidar_data.lidar_type] = str(lidar_data.relative_path)

        elif self._dataset_converter_config.lidar_store_option in ["laz_binary", "draco_binary"]:
            lidar_pcs_dict: Dict[LiDARType, np.ndarray] = {}

            # 1. Load point clouds from files
            for lidar_data in lidars:
                if lidar_data.has_point_cloud:
                    lidar_pcs_dict[lidar_data.lidar_type] = lidar_data.point_cloud
                elif lidar_data.has_file_path:
                    lidar_pcs_dict.update(
                        load_lidar_pcs_from_file(
                            lidar_data.relative_path,
                            self._log_metadata,
                            lidar_data.iteration,
                            lidar_data.dataset_root,
                        )
                    )

            # 2. Compress the point clouds to bytes
            for lidar_type, point_cloud in lidar_pcs_dict.items():
                lidar_metadata = self._log_metadata.lidar_metadata[lidar_type]
                binary: Optional[bytes] = None
                if self._dataset_converter_config.lidar_store_option == "draco_binary":
                    binary = encode_lidar_pc_as_draco_binary(point_cloud, lidar_metadata)
                elif self._dataset_converter_config.lidar_store_option == "laz_binary":
                    binary = encode_lidar_pc_as_laz_binary(point_cloud, lidar_metadata)
                else:
                    raise NotImplementedError(
                        f"Unsupported LiDAR store option: {self._dataset_converter_config.lidar_store_option}"
                    )
                lidar_data_dict[lidar_type] = binary

        return lidar_data_dict

    def _prepare_camera_data_dict(
        self, cameras: List[CameraData], store_option: Literal["path", "jpeg_binary", "png_binary", "mp4"]
    ) -> Dict[PinholeCameraType, Union[str, bytes]]:
        """Helper function to prepare camera data dictionary for the target storage option.

        :param cameras: List of CameraData objects to be processed.
        :param store_option: The storage option for camera data, either "path" or "binary".
        :raises NotImplementedError: If the storage option is not supported.
        :raises NotImplementedError: If the camera data does not support the specified storage option.
        :return: Dictionary mapping PinholeCameraType to either file path (str) or binary data (bytes) depending on storage option.
        """
        camera_data_dict: Dict[PinholeCameraType, Union[str, int, bytes]] = {}

        for camera_data in cameras:
            if store_option == "path":
                if camera_data.has_file_path:
                    camera_data_dict[camera_data.camera_type] = str(camera_data.relative_path)
                else:
                    raise NotImplementedError("Only file path storage is supported for camera data.")
            elif store_option == "jpeg_binary":
                camera_data_dict[camera_data.camera_type] = _get_jpeg_binary_from_camera_data(camera_data)
            elif store_option == "png_binary":
                camera_data_dict[camera_data.camera_type] = _get_png_binary_from_camera_data(camera_data)
            elif store_option == "mp4":
                camera_name = camera_data.camera_type.serialize()
                if camera_name not in self._pinhole_mp4_writers:
                    mp4_path = (
                        self._sensors_root
                        / self._log_metadata.split
                        / self._log_metadata.log_name
                        / f"{camera_name}.mp4"
                    )
                    frame_interval = self._log_metadata.timestep_seconds
                    self._pinhole_mp4_writers[camera_name] = MP4Writer(mp4_path, fps=1 / frame_interval)

                image = _get_numpy_image_from_camera_data(camera_data)
                camera_data_dict[camera_data.camera_type] = self._pinhole_mp4_writers[camera_name].write_frame(image)

            else:
                raise NotImplementedError(f"Unsupported camera store option: {store_option}")

        return camera_data_dict


def _get_jpeg_binary_from_camera_data(camera_data: CameraData) -> Optional[bytes]:
    jpeg_binary: Optional[bytes] = None

    if camera_data.has_jpeg_binary:
        jpeg_binary = camera_data.jpeg_binary
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path
        jpeg_binary = load_jpeg_binary_from_jpeg_file(absolute_path)
    elif camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path
        numpy_image = load_image_from_png_file(absolute_path)
        jpeg_binary = encode_image_as_jpeg_binary(numpy_image)
    elif camera_data.has_numpy_image:
        jpeg_binary = encode_image_as_jpeg_binary(camera_data.numpy_image)
    else:
        raise NotImplementedError("Camera data must provide jpeg_binary, numpy_image, or file path for binary storage.")

    assert jpeg_binary is not None
    return jpeg_binary


def _get_png_binary_from_camera_data(camera_data: CameraData) -> Optional[bytes]:
    png_binary: Optional[bytes] = None

    if camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path
        png_binary = load_png_binary_from_png_file(absolute_path)
    elif camera_data.has_numpy_image:
        png_binary = encode_image_as_png_binary(camera_data.numpy_image)
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path
        numpy_image = load_image_from_jpeg_file(absolute_path)
        png_binary = encode_image_as_png_binary(numpy_image)

    elif camera_data.has_jpeg_binary:
        numpy_image = decode_image_from_jpeg_binary(camera_data.jpeg_binary)
        png_binary = encode_image_as_png_binary(numpy_image)
    else:
        raise NotImplementedError("Camera data must provide png_binary, numpy_image, or file path for binary storage.")

    assert png_binary is not None
    return png_binary


def _get_numpy_image_from_camera_data(camera_data: CameraData) -> Optional[np.ndarray]:
    numpy_image: Optional[np.ndarray] = None

    if camera_data.has_numpy_image:
        numpy_image = camera_data.numpy_image
    elif camera_data.has_jpeg_binary:
        numpy_image = decode_image_from_jpeg_binary(camera_data.jpeg_binary)
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path
        numpy_image = load_image_from_jpeg_file(absolute_path)
    elif camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path
        numpy_image = load_image_from_png_file(absolute_path)
    else:
        raise NotImplementedError("Camera data must provide numpy_image, jpeg_binary, or file path for numpy image.")

    assert numpy_image is not None
    return numpy_image
