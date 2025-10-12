from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pyarrow as pa

from d123.common.utils.uuid import create_deterministic_uuid
from d123.conversion.abstract_dataset_converter import AbstractLogWriter, DatasetConverterConfig
from d123.datatypes.detections.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from d123.datatypes.scene.arrow.utils.arrow_metadata_utils import add_log_metadata_to_arrow_schema
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType
from d123.datatypes.sensors.lidar.lidar import LiDARType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import EgoStateSE3, EgoStateSE3Index
from d123.geometry import BoundingBoxSE3Index, StateSE3, StateSE3Index, Vector3DIndex


class ArrowLogWriter(AbstractLogWriter):

    def __init__(
        self,
        logs_root: Union[str, Path],
        compression: Optional[Literal["lz4", "zstd"]] = None,
        compression_level: Optional[int] = None,
    ) -> None:

        self._logs_root = Path(logs_root)
        self._compression = compression
        self._compression_level = compression_level

        # Loaded during .reset() and cleared during .close()
        self._dataset_converter_config: Optional[DatasetConverterConfig] = None
        self._log_metadata: Optional[LogMetadata] = None
        self._schema: Optional[LogMetadata] = None
        self._source: Optional[pa.NativeFile] = None
        self._record_batch_writer: Optional[pa.ipc.RecordBatchWriter] = None

    def reset(self, dataset_converter_config: DatasetConverterConfig, log_metadata: LogMetadata) -> bool:

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
                pa.Codec(self._compression, compression_level=self._compression_level)
                if self._compression is not None
                else None
            )

            options = pa.ipc.IpcWriteOptions(compression=compression)
            self._source = pa.OSFile(str(sink_log_path), "wb")
            self._record_batch_writer = pa.ipc.new_file(self._source, schema=self._schema, options=options)

        return log_needs_writing

    def write(
        self,
        timestamp: TimePoint,
        ego_state: Optional[EgoStateSE3] = None,
        box_detections: Optional[BoxDetectionWrapper] = None,
        traffic_lights: Optional[TrafficLightDetectionWrapper] = None,
        cameras: Optional[Dict[PinholeCameraType, Tuple[Any, ...]]] = None,
        lidars: Optional[Dict[LiDARType, Any]] = None,
        scenario_tags: Optional[List[str]] = None,
        route_lane_group_ids: Optional[List[int]] = None,
    ) -> None:

        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        assert self._schema is not None, "Log writer is not initialized."
        assert self._record_batch_writer is not None, "Log writer is not initialized."
        assert self._source is not None, "Log writer is not initialized."

        record_batch_data = {
            "uuid": [
                create_deterministic_uuid(
                    split=self._log_metadata.split,
                    log_name=self._log_metadata.log_name,
                    timestamp_us=timestamp.time_us,
                ).bytes
            ],
            "timestamp": [timestamp.time_us],
        }

        # --------------------------------------------------------------------------------------------------------------
        # Ego State
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_ego:
            assert ego_state is not None, "Ego state is required but not provided."
            record_batch_data["ego_state"] = [ego_state.array]

        # --------------------------------------------------------------------------------------------------------------
        # Box Detections
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_box_detections:
            assert box_detections is not None, "Box detections are required but not provided."
            # TODO: Figure out more elegant way without for-loops.

            # Accumulate box detection data
            box_detection_state = []
            box_detection_velocity = []
            box_detection_token = []
            box_detection_type = []

            for box_detection in box_detections:
                box_detection_state.append(box_detection.bounding_box.array)
                box_detection_velocity.append(box_detection.velocity.array)
                box_detection_token.append(box_detection.metadata.track_token)
                box_detection_type.append(int(box_detection.metadata.detection_type))

            # Add to record batch data
            record_batch_data["box_detection_state"] = [box_detection_state]
            record_batch_data["box_detection_velocity"] = [box_detection_velocity]
            record_batch_data["box_detection_token"] = [box_detection_token]
            record_batch_data["box_detection_type"] = [box_detection_type]

        # --------------------------------------------------------------------------------------------------------------
        # Traffic Lights
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_traffic_lights:
            assert traffic_lights is not None, "Traffic light detections are required but not provided."
            # TODO: Figure out more elegant way without for-loops.

            # Accumulate traffic light data
            traffic_light_ids = []
            traffic_light_types = []

            for traffic_light in traffic_lights:
                traffic_light_ids.append(traffic_light.lane_id)
                traffic_light_types.append(int(traffic_light.status))

            # Add to record batch data
            record_batch_data["traffic_light_ids"] = [traffic_light_ids]
            record_batch_data["traffic_light_types"] = [traffic_light_types]

        # --------------------------------------------------------------------------------------------------------------
        # Cameras
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_cameras:
            assert cameras is not None, "Camera data is required but not provided."
            provided_cameras = set(cameras.keys())
            expected_cameras = set(self._log_metadata.camera_metadata.keys())
            for camera_type in expected_cameras:
                camera_name = camera_type.serialize()

                # NOTE: Missing cameras are allowed, e.g., for synchronization mismatches.
                # In this case, we write None/null to the arrow table.
                camera_data: Optional[Any] = None
                camera_pose: Optional[StateSE3] = None
                if camera_type in provided_cameras:
                    camera_data, camera_pose = cameras[camera_type]

                record_batch_data[f"{camera_name}_data"] = [camera_data]
                record_batch_data[f"{camera_name}_extrinsic"] = [camera_pose.array if camera_pose else None]

        # --------------------------------------------------------------------------------------------------------------
        # LiDARs
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_lidars:
            assert lidars is not None, "LiDAR data is required but not provided."
            provided_lidars = set(lidars.keys())
            expected_lidars = set(self._log_metadata.lidar_metadata.keys())
            for lidar_type in expected_lidars:
                lidar_name = lidar_type.serialize()

                # NOTE: Missing LiDARs are allowed, similar to cameras
                # In this case, we write None/null to the arrow table.
                lidar_data: Optional[Any] = None
                if lidar_type in provided_lidars:
                    lidar_data = lidars[lidar_type]
                record_batch_data[f"{lidar_name}_data"] = [lidar_data]

        # --------------------------------------------------------------------------------------------------------------
        # Miscellaneous (Scenario Tags / Route)
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_scenario_tags:
            assert scenario_tags is not None, "Scenario tags are required but not provided."
            record_batch_data["scenario_tags"] = [scenario_tags]

        if self._dataset_converter_config.include_route:
            assert route_lane_group_ids is not None, "Route lane group IDs are required but not provided."
            record_batch_data["route_lane_group_ids"] = [route_lane_group_ids]

        record_batch = pa.record_batch(record_batch_data, schema=self._schema)
        self._record_batch_writer.write_batch(record_batch)

    def close(self) -> None:
        if self._record_batch_writer is not None:
            self._record_batch_writer.close()
            self._record_batch_writer: Optional[pa.ipc.RecordBatchWriter] = None

        if self._source is not None:
            self._source.close()
            self._source: Optional[pa.NativeFile] = None

        self._dataset_converter_config: Optional[DatasetConverterConfig] = None
        self._log_metadata: Optional[LogMetadata] = None
        self._schema: Optional[LogMetadata] = None

    @staticmethod
    def _build_schema(dataset_converter_config: DatasetConverterConfig, log_metadata: LogMetadata) -> pa.Schema:

        schema_list: List[Tuple[str, pa.DataType]] = [
            ("uuid", pa.uuid()),
            ("timestamp", pa.int64()),
        ]

        # --------------------------------------------------------------------------------------------------------------
        # Ego State
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_ego:
            schema_list.extend(
                [
                    ("ego_state", pa.list_(pa.float64(), len(EgoStateSE3Index))),
                ]
            )

        # --------------------------------------------------------------------------------------------------------------
        # Box Detections
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_box_detections:
            schema_list.extend(
                [
                    ("box_detection_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                    ("box_detection_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                    ("box_detection_token", pa.list_(pa.string())),
                    ("box_detection_type", pa.list_(pa.int16())),
                ]
            )

        # --------------------------------------------------------------------------------------------------------------
        # Traffic Lights
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_traffic_lights:
            schema_list.extend(
                [
                    ("traffic_light_ids", pa.list_(pa.int64())),
                    ("traffic_light_types", pa.list_(pa.int16())),
                ]
            )

        # --------------------------------------------------------------------------------------------------------------
        # Cameras
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_cameras:
            for camera_type in log_metadata.camera_metadata.keys():
                camera_name = camera_type.serialize()

                # Depending on the storage option, define the schema for camera data
                if dataset_converter_config.camera_store_option == "path":
                    schema_list.append((f"{camera_name}_data", pa.string()))

                elif dataset_converter_config.camera_store_option == "binary":
                    schema_list.append((f"{camera_name}_data", pa.binary()))

                # Add camera pose
                schema_list.append((f"{camera_name}_extrinsic", pa.list_(pa.float64(), len(StateSE3Index))))

        # --------------------------------------------------------------------------------------------------------------
        # LiDARs
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_lidars:
            for lidar_type in log_metadata.lidar_metadata.keys():
                lidar_name = lidar_type.serialize()

                # Depending on the storage option, define the schema for LiDAR data
                if dataset_converter_config.lidar_store_option == "path":
                    schema_list.append((f"{lidar_name}_data", pa.string()))

                elif dataset_converter_config.lidar_store_option == "binary":
                    schema_list.append((f"{lidar_name}_data", pa.binary()))

        # --------------------------------------------------------------------------------------------------------------
        # Miscellaneous (Scenario Tags / Route)
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_scenario_tags:
            schema_list.append(("scenario_tags", pa.list_(pa.string())))

        if dataset_converter_config.include_route:
            schema_list.append(("route_lane_group_ids", pa.list_(pa.int64())))

        return add_log_metadata_to_arrow_schema(pa.schema(schema_list), log_metadata)
