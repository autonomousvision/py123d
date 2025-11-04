from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import pyarrow as pa

from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.conversion.abstract_dataset_converter import AbstractLogWriter, DatasetConverterConfig
from py123d.conversion.log_writer.abstract_log_writer import LiDARData
from py123d.conversion.sensor_io.lidar.draco_lidar_io import encode_lidar_pc_as_draco_binary
from py123d.conversion.sensor_io.lidar.file_lidar_io import load_lidar_pcs_from_file
from py123d.conversion.sensor_io.lidar.laz_lidar_io import encode_lidar_pc_as_laz_binary
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.scene.arrow.utils.arrow_metadata_utils import add_log_metadata_to_arrow_schema
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3, EgoStateSE3Index
from py123d.geometry import BoundingBoxSE3Index, StateSE3, StateSE3Index, Vector3DIndex


class ArrowLogWriter(AbstractLogWriter):

    def __init__(
        self,
        logs_root: Union[str, Path],
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        lidar_compression: Optional[Literal["draco", "laz"]] = "draco",
    ) -> None:

        self._logs_root = Path(logs_root)
        self._ipc_compression = ipc_compression
        self._ipc_compression_level = ipc_compression_level
        self._lidar_compression = lidar_compression

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
                pa.Codec(self._ipc_compression, compression_level=self._ipc_compression_level)
                if self._ipc_compression is not None
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
        pinhole_cameras: Optional[Dict[PinholeCameraType, Tuple[Any, ...]]] = None,
        fisheye_mei_cameras: Optional[Dict[FisheyeMEICameraType, Tuple[Any, ...]]] = None,
        lidars: Optional[List[LiDARData]] = None,
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
                box_detection_velocity.append(box_detection.velocity.array)  # TODO: make optional
                box_detection_token.append(box_detection.metadata.track_token)
                box_detection_type.append(int(box_detection.metadata.box_detection_type))

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
        # Pinhole Cameras
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_pinhole_cameras:
            assert pinhole_cameras is not None, "Pinhole camera data is required but not provided."
            provided_pinhole_cameras = set(pinhole_cameras.keys())
            expected_pinhole_cameras = set(self._log_metadata.pinhole_camera_metadata.keys())
            for pinhole_camera_type in expected_pinhole_cameras:
                pinhole_camera_name = pinhole_camera_type.serialize()

                # NOTE @DanielDauner: Missing cameras are allowed, e.g., for synchronization mismatches.
                # In this case, we write None/null to the arrow table.
                pinhole_camera_data: Optional[Any] = None
                pinhole_camera_pose: Optional[StateSE3] = None
                if pinhole_camera_type in provided_pinhole_cameras:
                    pinhole_camera_data, pinhole_camera_pose = pinhole_cameras[pinhole_camera_type]

                # TODO: Refactor how camera data handed to the writer.
                # This should be combined with configurations to write to log, sensor_root, or sensor_root as mp4.
                if isinstance(pinhole_camera_data, Path) or isinstance(pinhole_camera_data, str):
                    pinhole_camera_data = str(pinhole_camera_data)
                elif isinstance(pinhole_camera_data, bytes):
                    pinhole_camera_data = pinhole_camera_data
                elif isinstance(pinhole_camera_data, np.ndarray):
                    _, encoded_img = cv2.imencode(".jpg", pinhole_camera_data)
                    pinhole_camera_data = encoded_img.tobytes()

                record_batch_data[f"{pinhole_camera_name}_data"] = [pinhole_camera_data]
                record_batch_data[f"{pinhole_camera_name}_extrinsic"] = [
                    pinhole_camera_pose.array if pinhole_camera_pose else None
                ]

        # --------------------------------------------------------------------------------------------------------------
        # Fisheye MEI Cameras
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_fisheye_mei_cameras:
            assert fisheye_mei_cameras is not None, "Fisheye MEI camera data is required but not provided."
            provided_fisheye_mei_cameras = set(fisheye_mei_cameras.keys())
            expected_fisheye_mei_cameras = set(self._log_metadata.fisheye_mei_camera_metadata.keys())
            for fisheye_mei_camera_type in expected_fisheye_mei_cameras:
                fisheye_mei_camera_name = fisheye_mei_camera_type.serialize()

                # NOTE @DanielDauner: Missing cameras are allowed, e.g., for synchronization mismatches.
                # In this case, we write None/null to the arrow table.
                fisheye_mei_camera_data: Optional[Any] = None
                fisheye_mei_camera_pose: Optional[StateSE3] = None
                if fisheye_mei_camera_type in provided_fisheye_mei_cameras:
                    fisheye_mei_camera_data, fisheye_mei_camera_pose = fisheye_mei_cameras[fisheye_mei_camera_type]

                # TODO: Refactor how camera data handed to the writer.
                # This should be combined with configurations to write to log, sensor_root, or sensor_root as mp4.
                if isinstance(fisheye_mei_camera_data, Path) or isinstance(fisheye_mei_camera_data, str):
                    fisheye_mei_camera_data = str(fisheye_mei_camera_data)
                elif isinstance(fisheye_mei_camera_data, bytes):
                    fisheye_mei_camera_data = fisheye_mei_camera_data
                elif isinstance(fisheye_mei_camera_data, np.ndarray):
                    _, encoded_img = cv2.imencode(".jpg", fisheye_mei_camera_data)
                    fisheye_mei_camera_data = encoded_img.tobytes()

                record_batch_data[f"{fisheye_mei_camera_name}_data"] = [fisheye_mei_camera_data]
                record_batch_data[f"{fisheye_mei_camera_name}_extrinsic"] = [
                    fisheye_mei_camera_pose.array if fisheye_mei_camera_pose else None
                ]

        # --------------------------------------------------------------------------------------------------------------
        # LiDARs
        # --------------------------------------------------------------------------------------------------------------
        if self._dataset_converter_config.include_lidars and len(self._log_metadata.lidar_metadata) > 0:
            assert lidars is not None, "LiDAR data is required but not provided."

            if self._dataset_converter_config.lidar_store_option == "path_merged":
                # NOTE @DanielDauner: The path_merged option is necessary for dataset, that natively store multiple
                # LiDAR point clouds in a single file. In this case, writing the file path several times is wasteful.
                # Instead, we store the file path once, and divide the point clouds during reading.
                assert len(lidars) == 1, "Exactly one LiDAR data must be provided for merged LiDAR storage."
                assert lidars[0].has_file_path, "LiDAR data must provide file path for merged LiDAR storage."
                merged_lidar_data: Optional[str] = str(lidars[0].relative_path)

                record_batch_data[f"{LiDARType.LIDAR_MERGED.serialize()}_data"] = [merged_lidar_data]

            else:
                # NOTE @DanielDauner: for "path" and "binary" options, we write each LiDAR in a separate column.
                # We currently assume that all lidars are provided at the same time step.
                # Theoretically, we could extend the store asynchronous LiDARs in the future by storing the lidar data
                # list as a dictionary, list or struct-like object in the columns.
                expected_lidars = set(self._log_metadata.lidar_metadata.keys())
                lidar_data_dict = self._prepare_lidar_data_dict(lidars)

                for lidar_type in expected_lidars:
                    lidar_name = lidar_type.serialize()
                    lidar_data: Optional[Union[str, bytes]] = lidar_data_dict.get(lidar_type, None)
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
        # Pinhole Cameras
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_pinhole_cameras:
            for pinhole_camera_type in log_metadata.pinhole_camera_metadata.keys():
                pinhole_camera_name = pinhole_camera_type.serialize()

                # Depending on the storage option, define the schema for camera data
                if dataset_converter_config.pinhole_camera_store_option == "path":
                    schema_list.append((f"{pinhole_camera_name}_data", pa.string()))

                elif dataset_converter_config.pinhole_camera_store_option == "binary":
                    schema_list.append((f"{pinhole_camera_name}_data", pa.binary()))

                elif dataset_converter_config.pinhole_camera_store_option == "mp4":
                    raise NotImplementedError("MP4 format is not yet supported, but planned for future releases.")

                # Add camera pose
                schema_list.append((f"{pinhole_camera_name}_extrinsic", pa.list_(pa.float64(), len(StateSE3Index))))

        # --------------------------------------------------------------------------------------------------------------
        # Fisheye MEI Cameras
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_fisheye_mei_cameras:
            for fisheye_mei_camera_type in log_metadata.fisheye_mei_camera_metadata.keys():
                fisheye_mei_camera_name = fisheye_mei_camera_type.serialize()

                # Depending on the storage option, define the schema for camera data
                if dataset_converter_config.fisheye_mei_camera_store_option == "path":
                    schema_list.append((f"{fisheye_mei_camera_name}_data", pa.string()))

                elif dataset_converter_config.fisheye_mei_camera_store_option == "binary":
                    schema_list.append((f"{fisheye_mei_camera_name}_data", pa.binary()))

                elif dataset_converter_config.fisheye_mei_camera_store_option == "mp4":
                    raise NotImplementedError("MP4 format is not yet supported, but planned for future releases.")

                # Add camera pose
                schema_list.append((f"{fisheye_mei_camera_name}_extrinsic", pa.list_(pa.float64(), len(StateSE3Index))))

        # --------------------------------------------------------------------------------------------------------------
        # LiDARs
        # --------------------------------------------------------------------------------------------------------------
        if dataset_converter_config.include_lidars and len(log_metadata.lidar_metadata) > 0:
            if dataset_converter_config.lidar_store_option == "path_merged":
                schema_list.append((f"{LiDARType.LIDAR_MERGED.serialize()}_data", pa.string()))
            else:
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

    def _prepare_lidar_data_dict(self, lidars: List[LiDARData]) -> Dict[LiDARType, Union[str, bytes]]:
        lidar_data_dict: Dict[LiDARType, Union[str, bytes]] = {}

        if self._dataset_converter_config.lidar_store_option == "path":
            for lidar_data in lidars:
                assert lidar_data.has_file_path, "LiDAR data must provide file path for path storage."
                lidar_data_dict[lidar_data.lidar_type] = str(lidar_data.relative_path)

        elif self._dataset_converter_config.lidar_store_option == "binary":
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
                if self._lidar_compression == "draco":
                    binary = encode_lidar_pc_as_draco_binary(point_cloud, lidar_metadata)
                elif self._lidar_compression == "laz":
                    binary = encode_lidar_pc_as_laz_binary(point_cloud, lidar_metadata)
                lidar_data_dict[lidar_type] = binary

        return lidar_data_dict
