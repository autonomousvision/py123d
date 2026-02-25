import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
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
    EGO_IMU_SE3_COLUMN,
    FISHEYE_CAMERA_DATA_COLUMN,
    FISHEYE_CAMERA_EXTRINSIC_COLUMN,
    FISHEYE_CAMERA_TIMESTAMP_COLUMN,
    LIDAR_PATH_COLUMN,
    LIDAR_POINT_CLOUD_COLUMN,
    LIDAR_POINT_CLOUD_FEATURE_COLUMN,
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
from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.conversion.abstract_dataset_converter import AbstractLogWriter, DatasetConverterConfig
from py123d.conversion.log_writer.abstract_log_writer import CameraData, LidarData
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
from py123d.conversion.sensor_io.lidar.draco_lidar_io import encode_point_cloud_3d_as_draco_binary
from py123d.conversion.sensor_io.lidar.ipc_lidar_io import (
    encode_point_cloud_3d_as_ipc_binary,
    encode_point_cloud_features_as_ipc_binary,
)
from py123d.conversion.sensor_io.lidar.laz_lidar_io import encode_point_cloud_3d_as_laz_binary
from py123d.conversion.sensor_io.lidar.path_lidar_io import load_point_cloud_data_from_path
from py123d.datatypes import (
    BoxDetectionSE3,
    BoxDetectionWrapper,
    DynamicStateSE3Index,
    EgoStateSE3,
    LidarID,
    LogMetadata,
    PinholeCameraID,
    Timestamp,
    TrafficLightDetectionWrapper,
)
from py123d.geometry import BoundingBoxSE3Index, PoseSE3, PoseSE3Index, Vector3DIndex


def _get_logs_root() -> Path:
    logs_root = get_dataset_paths().py123d_logs_root
    assert logs_root is not None, "PY123D_DATA_ROOT must be set."
    return logs_root


def _get_sensors_root() -> Path:
    sensors_root = get_dataset_paths().py123d_sensors_root
    assert sensors_root is not None, "PY123D_DATA_ROOT must be set."
    return sensors_root


def _camera_store_option_to_arrow_type(
    store_option: Literal["path", "jpeg_binary", "png_binary", "mp4"],
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


class ArrowLogWriter(AbstractLogWriter):
    """Log writer for Arrow-based logs. Writes log data to an Arrow IPC file format."""

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
        self._schema: Optional[pa.Schema] = None
        self._source: Optional[pa.NativeFile] = None
        self._record_batch_writer: Optional[pa.ipc.RecordBatchWriter] = None  # pyright: ignore[reportAttributeAccessIssue]
        self._pinhole_mp4_writers: Dict[str, MP4Writer] = {}
        self._fisheye_mei_mp4_writers: Dict[str, MP4Writer] = {}

    def reset(self, dataset_converter_config: DatasetConverterConfig, log_metadata: LogMetadata) -> bool:
        """Inherited, see superclass."""

        log_needs_writing: bool = False
        sink_log_path: Path = self._logs_root / log_metadata.split / f"{log_metadata.log_name}.arrow"

        # Check if the log file already exists or needs to be overwritten
        if not sink_log_path.exists() or dataset_converter_config.force_log_conversion:
            log_needs_writing = True

            # Delete the file if it exists (no error if it doesn't)
            sink_log_path.unlink(missing_ok=True)
            if not sink_log_path.parent.exists():
                sink_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Load config and metadata
            self._dataset_converter_config = dataset_converter_config
            self._log_metadata = log_metadata
            self._schema = self._build_schema(dataset_converter_config, log_metadata)

            # Initialize Arrow IPC writer, optionally with compression
            # NOTE @DanielDauner: I tried some compression settings, which did not lead to significant reductions.
            compression = (
                pa.Codec(self._ipc_compression, compression_level=self._ipc_compression_level)
                if self._ipc_compression is not None
                else None
            )

            options = pa.ipc.IpcWriteOptions(compression=compression)
            self._source = pa.OSFile(str(sink_log_path), "wb")
            self._record_batch_writer = pa.ipc.new_file(self._source, schema=self._schema, options=options)

            self._pinhole_mp4_writers = {}
            self._fisheye_mei_mp4_writers = {}

        return log_needs_writing

    def write(
        self,
        timestamp: Timestamp,
        uuid: Optional[uuid.UUID] = None,
        ego_state: Optional[EgoStateSE3] = None,
        box_detections: Optional[BoxDetectionWrapper] = None,
        traffic_lights: Optional[TrafficLightDetectionWrapper] = None,
        pinhole_cameras: Optional[List[CameraData]] = None,
        fisheye_mei_cameras: Optional[List[CameraData]] = None,
        lidar: Optional[LidarData] = None,
        scenario_tags: Optional[List[str]] = None,
        route_lane_group_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Inherited, see superclass."""

        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        assert self._schema is not None, "Log writer is not initialized."
        assert self._record_batch_writer is not None, "Log writer is not initialized."
        assert self._source is not None, "Log writer is not initialized."

        if uuid is None:
            uuid = create_deterministic_uuid(
                split=self._log_metadata.split,
                log_name=self._log_metadata.log_name,
                timestamp_us=timestamp.time_us,
            )
        record_batch_data = {UUID_COLUMN: [uuid.bytes], TIMESTAMP_US_COLUMN: [timestamp.time_us]}

        # --------------------------------------------------------------------------------------------------------------
        # Ego State
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_ego:
            assert ego_state is not None, "Ego state is required but not provided."
            record_batch_data[EGO_IMU_SE3_COLUMN] = [ego_state.imu_se3]
            record_batch_data[EGO_DYNAMIC_STATE_SE3_COLUMN] = [ego_state.dynamic_state_se3]

        # --------------------------------------------------------------------------------------------------------------
        # Box Detections
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_box_detections:
            assert box_detections is not None, "Box detections are required but not provided."

            # Accumulate box detection data
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

            # Add to record batch data
            record_batch_data[BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN] = [box_detection_state]
            record_batch_data[BOX_DETECTIONS_TOKEN_COLUMN] = [box_detection_token]
            record_batch_data[BOX_DETECTIONS_LABEL_COLUMN] = [box_detection_label]
            record_batch_data[BOX_DETECTIONS_VELOCITY_3D_COLUMN] = [box_detection_velocity]
            record_batch_data[BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN] = [box_detection_num_lidar_points]

        # --------------------------------------------------------------------------------------------------------------
        # Traffic Lights
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_traffic_lights:
            assert traffic_lights is not None, "Traffic light detections are required but not provided."

            # Accumulate traffic light data
            traffic_light_ids = []
            traffic_light_statuses = []

            for traffic_light in traffic_lights:
                traffic_light_ids.append(traffic_light.lane_id)
                traffic_light_statuses.append(int(traffic_light.status))

            # Add to record batch data
            record_batch_data[TRAFFIC_LIGHTS_LANE_ID_COLUMN] = [traffic_light_ids]
            record_batch_data[TRAFFIC_LIGHTS_STATUS_COLUMN] = [traffic_light_statuses]

        # --------------------------------------------------------------------------------------------------------------
        # Pinhole Cameras
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_pinhole_cameras:
            assert pinhole_cameras is not None, "Pinhole camera data is required but not provided."
            provided_pinhole_data = self._prepare_camera_data_dict(
                pinhole_cameras, self._dataset_converter_config.pinhole_camera_store_option
            )
            provided_pinhole_extrinsics = {
                camera_data.camera_id: camera_data.extrinsic for camera_data in pinhole_cameras
            }
            provided_pinhole_timestamps = {
                camera_data.camera_id: camera_data.timestamp for camera_data in pinhole_cameras
            }
            expected_pinhole_cameras = set(self._log_metadata.pinhole_camera_metadata.keys())

            for pinhole_camera_type in expected_pinhole_cameras:
                pinhole_camera_name = pinhole_camera_type.serialize()

                # NOTE @DanielDauner: Missing cameras are allowed, e.g., for synchronization mismatches.
                # In this case, we write None/null to the arrow table.
                # Theoretically, we could extend the store asynchronous cameras in the future by storing the
                # camera data as a dictionary, list or struct-like object in the columns.
                pinhole_camera_data: Optional[Any] = None
                pinhole_camera_pose: Optional[PoseSE3] = None
                pinhole_camera_timestamp: Optional[Timestamp] = None
                if pinhole_camera_type in provided_pinhole_data:
                    pinhole_camera_data = provided_pinhole_data[pinhole_camera_type]
                    pinhole_camera_pose = provided_pinhole_extrinsics[pinhole_camera_type]
                    pinhole_camera_timestamp = provided_pinhole_timestamps[pinhole_camera_type]

                record_batch_data[PINHOLE_CAMERA_DATA_COLUMN(pinhole_camera_name)] = [pinhole_camera_data]
                record_batch_data[PINHOLE_CAMERA_EXTRINSIC_COLUMN(pinhole_camera_name)] = [
                    pinhole_camera_pose.array if pinhole_camera_pose is not None else None
                ]
                record_batch_data[PINHOLE_CAMERA_TIMESTAMP_COLUMN(pinhole_camera_name)] = [
                    pinhole_camera_timestamp.time_us if pinhole_camera_timestamp is not None else None
                ]

        # --------------------------------------------------------------------------------------------------------------
        # Fisheye MEI Cameras
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_fisheye_mei_cameras:
            assert fisheye_mei_cameras is not None, "Fisheye MEI camera data is required but not provided."
            provided_fisheye_mei_data = self._prepare_camera_data_dict(
                fisheye_mei_cameras, self._dataset_converter_config.fisheye_mei_camera_store_option
            )
            provided_fisheye_mei_extrinsics = {
                camera_data.camera_id: camera_data.extrinsic for camera_data in fisheye_mei_cameras
            }
            provided_fisheye_mei_timestamps = {
                camera_data.camera_id: camera_data.timestamp for camera_data in fisheye_mei_cameras
            }
            expected_fisheye_mei_cameras = set(self._log_metadata.fisheye_mei_camera_metadata.keys())

            for fisheye_mei_camera_type in expected_fisheye_mei_cameras:
                fisheye_mei_camera_name = fisheye_mei_camera_type.serialize()

                # NOTE @DanielDauner: Missing cameras are allowed, e.g., for synchronization mismatches.
                # In this case, we write None/null to the arrow table.
                fisheye_mei_camera_data: Optional[Any] = None
                fisheye_mei_camera_pose: Optional[PoseSE3] = None
                fisheye_mei_camera_timestamp: Optional[Timestamp] = None
                if fisheye_mei_camera_type in provided_fisheye_mei_data:
                    fisheye_mei_camera_data = provided_fisheye_mei_data[fisheye_mei_camera_type]
                    fisheye_mei_camera_pose = provided_fisheye_mei_extrinsics[fisheye_mei_camera_type]
                    fisheye_mei_camera_timestamp = provided_fisheye_mei_timestamps[fisheye_mei_camera_type]

                record_batch_data[FISHEYE_CAMERA_DATA_COLUMN(fisheye_mei_camera_name)] = [fisheye_mei_camera_data]
                record_batch_data[FISHEYE_CAMERA_EXTRINSIC_COLUMN(fisheye_mei_camera_name)] = [
                    fisheye_mei_camera_pose.array if fisheye_mei_camera_pose is not None else None
                ]
                record_batch_data[FISHEYE_CAMERA_TIMESTAMP_COLUMN(fisheye_mei_camera_name)] = [
                    fisheye_mei_camera_timestamp.time_us if fisheye_mei_camera_timestamp is not None else None
                ]

        # --------------------------------------------------------------------------------------------------------------
        # Lidars
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_lidars and len(self._log_metadata.lidar_metadata) > 0:
            if self._dataset_converter_config.lidar_store_option == "path":
                lidar_path: Optional[str] = None
                if lidar is not None:
                    assert lidar.has_file_path
                    lidar_path = str(lidar.relative_path)
                record_batch_data[LIDAR_PATH_COLUMN(LidarID.LIDAR_MERGED.serialize())] = [lidar_path]
            elif self._dataset_converter_config.lidar_store_option == "binary":
                lidar_name = LidarID.LIDAR_MERGED.serialize()
                lidar_point_cloud_binary: Optional[bytes] = None
                lidar_point_cloud_features: Optional[bytes] = None
                if lidar is not None:
                    lidar_point_cloud_binary, lidar_point_cloud_features = self._prepare_lidar_data_dict(lidar)
                record_batch_data[LIDAR_POINT_CLOUD_COLUMN(lidar_name)] = [lidar_point_cloud_binary]
                record_batch_data[LIDAR_POINT_CLOUD_FEATURE_COLUMN(lidar_name)] = [lidar_point_cloud_features]
            else:
                raise ValueError(f"Unsupported Lidar store option: {self._dataset_converter_config.lidar_store_option}")

        # --------------------------------------------------------------------------------------------------------------
        # Miscellaneous (Scenario Tags / Route)
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_scenario_tags:
            assert scenario_tags is not None, "Scenario tags are required but not provided."
            record_batch_data[SCENARIO_TAGS_COLUMN] = [scenario_tags]

        if self._dataset_converter_config.include_route:
            assert route_lane_group_ids is not None, "Route lane group IDs are required but not provided."
            record_batch_data[ROUTE_LANE_GROUP_IDS_COLUMN] = [route_lane_group_ids]

        record_batch = pa.record_batch(record_batch_data, schema=self._schema)
        self._record_batch_writer.write_batch(record_batch)

    def close(self) -> None:
        """Inherited, see superclass."""
        if self._record_batch_writer is not None:
            self._record_batch_writer.close()
            self._record_batch_writer: Optional[pa.ipc.RecordBatchWriter] = None  # type: ignore

        if self._source is not None:
            self._source.close()
            self._source: Optional[pa.NativeFile] = None

        self._dataset_converter_config: Optional[DatasetConverterConfig] = None
        self._log_metadata: Optional[LogMetadata] = None
        self._schema: Optional[pa.Schema] = None

        for mp4_writer in self._pinhole_mp4_writers.values():
            mp4_writer.close()
        self._pinhole_mp4_writers = {}
        for mp4_writer in self._fisheye_mei_mp4_writers.values():
            mp4_writer.close()
        self._fisheye_mei_mp4_writers = {}

    @staticmethod
    def _build_schema(dataset_converter_config: DatasetConverterConfig, log_metadata: LogMetadata) -> pa.Schema:
        """Builds the schema for the Arrow table, specifying datatypes and modalities to be stored.

        :param dataset_converter_config: The dataset converter configuration.
        :param log_metadata: The metadata for the log.
        :return: The Arrow schema object.
        """

        schema_list: List[Tuple[str, pa.DataType]] = [
            (UUID_COLUMN, _get_uuid_arrow_type()),
            (TIMESTAMP_US_COLUMN, pa.int64()),
        ]

        # --------------------------------------------------------------------------------------------------------------
        # Ego State
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_ego:
            schema_list.extend(
                [
                    (EGO_IMU_SE3_COLUMN, pa.list_(pa.float64(), len(PoseSE3Index))),
                    (EGO_DYNAMIC_STATE_SE3_COLUMN, pa.list_(pa.float64(), len(DynamicStateSE3Index))),
                ]
            )

        # --------------------------------------------------------------------------------------------------------------
        # Box Detections
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_box_detections:
            schema_list.extend(
                [
                    (
                        BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN,
                        pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index))),
                    ),
                    (
                        BOX_DETECTIONS_TOKEN_COLUMN,
                        pa.list_(pa.string()),
                    ),
                    (
                        BOX_DETECTIONS_LABEL_COLUMN,
                        pa.list_(pa.int16()),
                    ),
                    (
                        BOX_DETECTIONS_VELOCITY_3D_COLUMN,
                        pa.list_(pa.list_(pa.float64(), len(Vector3DIndex))),
                    ),
                    (
                        BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN,
                        pa.list_(pa.int64()),
                    ),
                ]
            )

        # --------------------------------------------------------------------------------------------------------------
        # Traffic Lights
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_traffic_lights:
            schema_list.extend(
                [
                    (TRAFFIC_LIGHTS_LANE_ID_COLUMN, pa.list_(pa.int64())),
                    (TRAFFIC_LIGHTS_STATUS_COLUMN, pa.list_(pa.int16())),
                ]
            )

        # --------------------------------------------------------------------------------------------------------------
        # Pinhole Cameras
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_pinhole_cameras:
            for pinhole_camera_type in log_metadata.pinhole_camera_metadata.keys():
                pinhole_camera_name = pinhole_camera_type.serialize()
                schema_list.extend(
                    [
                        (
                            PINHOLE_CAMERA_DATA_COLUMN(pinhole_camera_name),
                            _camera_store_option_to_arrow_type(dataset_converter_config.pinhole_camera_store_option),
                        ),
                        (
                            PINHOLE_CAMERA_EXTRINSIC_COLUMN(pinhole_camera_name),
                            pa.list_(pa.float64(), len(PoseSE3Index)),
                        ),
                        (
                            PINHOLE_CAMERA_TIMESTAMP_COLUMN(pinhole_camera_name),
                            pa.int64(),
                        ),
                    ]
                )

        # --------------------------------------------------------------------------------------------------------------
        # Fisheye MEI Cameras
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_fisheye_mei_cameras:
            for fisheye_mei_camera_type in log_metadata.fisheye_mei_camera_metadata.keys():
                fisheye_mei_camera_name = fisheye_mei_camera_type.serialize()
                schema_list.extend(
                    [
                        (
                            FISHEYE_CAMERA_DATA_COLUMN(fisheye_mei_camera_name),
                            _camera_store_option_to_arrow_type(
                                dataset_converter_config.fisheye_mei_camera_store_option
                            ),
                        ),
                        (
                            FISHEYE_CAMERA_EXTRINSIC_COLUMN(fisheye_mei_camera_name),
                            pa.list_(pa.float64(), len(PoseSE3Index)),
                        ),
                        (
                            FISHEYE_CAMERA_TIMESTAMP_COLUMN(fisheye_mei_camera_name),
                            pa.int64(),
                        ),
                    ]
                )

        # --------------------------------------------------------------------------------------------------------------
        # Lidars
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_lidars and len(log_metadata.lidar_metadata) > 0:
            # NOTE @DanielDauner: We now store the lidar merged by default.

            if dataset_converter_config.lidar_store_option == "path":
                schema_list.append(
                    (
                        LIDAR_PATH_COLUMN(LidarID.LIDAR_MERGED.serialize()),
                        pa.string(),
                    )
                )
            elif dataset_converter_config.lidar_store_option == "binary":
                schema_list.append(
                    (
                        LIDAR_POINT_CLOUD_COLUMN(LidarID.LIDAR_MERGED.serialize()),
                        pa.binary(),
                    )
                )
                if dataset_converter_config.lidar_point_feature_codec is not None:
                    # If a point feature codec is specified, we also store the point features as binary data.
                    schema_list.append(
                        (
                            LIDAR_POINT_CLOUD_FEATURE_COLUMN(LidarID.LIDAR_MERGED.serialize()),
                            pa.binary(),
                        )
                    )
            else:
                raise NotImplementedError(
                    f"Unsupported Lidar store option: {dataset_converter_config.lidar_store_option}"
                )

        # --------------------------------------------------------------------------------------------------------------
        # Miscellaneous (Scenario Tags / Route)
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_scenario_tags:
            schema_list.append((SCENARIO_TAGS_COLUMN, pa.list_(pa.string())))

        if dataset_converter_config.include_route:
            schema_list.append((ROUTE_LANE_GROUP_IDS_COLUMN, pa.list_(pa.int64())))

        return add_log_metadata_to_arrow_schema(pa.schema(schema_list), log_metadata)

    def _prepare_lidar_data_dict(self, lidar_data: LidarData) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Load and/or encodes the lidar data in binary for the point cloud and additional features.

        :param lidar_data: Helper class to reference the lidar observation.
        :return: Tuple of (point_cloud_binary, point_cloud_features_binary)
        """
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."

        # 1. Load point cloud and point features
        point_cloud_3d: Optional[npt.NDArray] = None
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
        if lidar_data.has_point_cloud_3d:
            point_cloud_3d = lidar_data.point_cloud_3d
            point_cloud_features = lidar_data.point_cloud_features
        elif lidar_data.has_file_path:
            point_cloud_3d, point_cloud_features = load_point_cloud_data_from_path(
                lidar_data.relative_path,  # type: ignore
                self._log_metadata,
                lidar_data.iteration,
                lidar_data.dataset_root,
            )
        else:
            raise ValueError("Lidar data must provide either point cloud data or a file path.")

        # 2. Compress point clouds with target codec
        point_cloud_3d_output: Optional[bytes] = None
        if point_cloud_3d is not None:
            if self._dataset_converter_config.lidar_point_cloud_codec == "draco":
                point_cloud_3d_output = encode_point_cloud_3d_as_draco_binary(point_cloud_3d)
            elif self._dataset_converter_config.lidar_point_cloud_codec == "laz":
                point_cloud_3d_output = encode_point_cloud_3d_as_laz_binary(point_cloud_3d)
            elif self._dataset_converter_config.lidar_point_cloud_codec == "ipc":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec=None)
            elif self._dataset_converter_config.lidar_point_cloud_codec == "ipc_zstd":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec="zstd")
            elif self._dataset_converter_config.lidar_point_cloud_codec == "ipc_lz4":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec="lz4")
            else:
                raise NotImplementedError(
                    f"Unsupported Lidar point cloud codec: {self._dataset_converter_config.lidar_point_cloud_codec}"
                )

        # 3. Compress point cloud features with target codec, if specified
        point_cloud_feature_output: Optional[bytes] = None
        if self._dataset_converter_config.lidar_point_feature_codec is not None and point_cloud_features is not None:
            if self._dataset_converter_config.lidar_point_feature_codec == "ipc":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(point_cloud_features, codec=None)
            elif self._dataset_converter_config.lidar_point_feature_codec == "ipc_zstd":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="zstd"
                )
            elif self._dataset_converter_config.lidar_point_feature_codec == "ipc_lz4":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="lz4"
                )
            else:
                raise NotImplementedError(
                    f"Unsupported Lidar point feature codec: {self._dataset_converter_config.lidar_point_feature_codec}"
                )

        return point_cloud_3d_output, point_cloud_feature_output

    def _prepare_camera_data_dict(
        self,
        cameras: List[CameraData],
        store_option: Literal["path", "jpeg_binary", "png_binary", "mp4"],
    ) -> Dict[PinholeCameraID, Union[str, bytes]]:
        """Helper function to prepare camera data dictionary for the target storage option.

        :param cameras: List of CameraData objects to be processed.
        :param store_option: The storage option for camera data, either "path" or "binary".
        :raises NotImplementedError: If the storage option is not supported.
        :raises NotImplementedError: If the camera data does not support the specified storage option.
        :return: Dictionary mapping PinholeCameraID to either file path (str) or binary data (bytes) depending on storage option.
        """
        camera_data_dict: Dict[PinholeCameraID, Union[str, int, bytes]] = {}

        for camera_data in cameras:
            if store_option == "path":
                if camera_data.has_file_path:
                    camera_data_dict[camera_data.camera_id] = str(camera_data.relative_path)
                else:
                    raise NotImplementedError("Only file path storage is supported for camera data.")
            elif store_option == "jpeg_binary":
                camera_data_dict[camera_data.camera_id] = _get_jpeg_binary_from_camera_data(camera_data)
            elif store_option == "png_binary":
                camera_data_dict[camera_data.camera_id] = _get_png_binary_from_camera_data(camera_data)
            elif store_option == "mp4":
                camera_name = camera_data.camera_id.serialize()
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
                camera_data_dict[camera_data.camera_id] = self._pinhole_mp4_writers[camera_name].write_frame(image)

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
