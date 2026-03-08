import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pyarrow as pa

from py123d.api.scene.abstract_log_writer import AbstractLogWriter
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Writer
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowFisheyeMEICameraWriter, ArrowPinholeCameraWriter
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityWriter
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Writer
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarWriter
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections_writer import ArrowTrafficLightDetectionsWriter
from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.api.utils.arrow_schema import SYNC
from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.datatypes import LidarID, LogMetadata, PinholeCameraMetadata, Timestamp
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
from py123d.datatypes.metadata.base_metadata import BaseModalityMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.datatypes.vehicle_state.ego_metadata import EgoStateSE3Metadata
from py123d.parser.abstract_dataset_parser import FrameData
from py123d.parser.dataset_converter_config import DatasetConverterConfig


@dataclass
class ArrowLogWriterState:
    log_dir: Path
    log_metadata: LogMetadata
    deferred_sync: bool = False
    modality_writers: Dict[str, BaseModalityWriter] = field(default_factory=dict)
    # For deferred sync: (row_index, timestamp_us) per modality_name
    timestamp_log: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))


class ArrowLogWriter(AbstractLogWriter):
    def __init__(
        self,
        dataset_converter_config: DatasetConverterConfig,
        logs_root: Union[str, Path],
        sensors_root: Union[str, Path],
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        """Initializes the :class:`ArrowLogWriter`.

        :param dataset_converter_config: The dataset converter configuration.
        :param logs_root: The root directory for logs.
        :param sensors_root: The root directory for sensors (e.g. MP4 video files).
        :param ipc_compression: The IPC compression method, defaults to None.
        :param ipc_compression_level: The IPC compression level, defaults to None.
        """
        self._dataset_converter_config = dataset_converter_config
        self._logs_root = Path(logs_root)
        self._sensors_root = Path(sensors_root)
        self._ipc_compression: Optional[Literal["lz4", "zstd"]] = ipc_compression
        self._ipc_compression_level: Optional[int] = ipc_compression_level

        self._state: Optional[ArrowLogWriterState] = None

    # ------------------------------------------------------------------------------------------------------------------
    # Writer lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def _close_writers(self) -> None:
        """Close all open modality writers."""
        if self._state is not None:
            for writer in self._state.modality_writers.values():
                writer.close()
            self._state.modality_writers.clear()

    def reset(
        self,
        log_metadata: LogMetadata,
        modality_metadatas: List[BaseModalityMetadata],
        deferred_sync: bool = False,
    ) -> bool:
        """Prepare the writer for a new log. Returns True if the log needs writing.

        :param log_metadata: Metadata for the log to write.
        :param modality_metadatas: List of modality metadata instances describing each modality.
        :param deferred_sync: If True, the sync table is built at close() from buffered timestamps.
        """
        assert self._state is None, "Log writer is already initialized. Call close() before reset()."

        log_dir: Path = self._logs_root / log_metadata.split / log_metadata.log_name
        sync_file_path = log_dir / "sync.arrow"

        if not sync_file_path.exists() or self._dataset_converter_config.force_log_conversion:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._state = ArrowLogWriterState(
                log_dir=log_dir,
                log_metadata=log_metadata,
                deferred_sync=deferred_sync,
            )
            for metadata in modality_metadatas:
                self._init_modality_writer(metadata)
            return True

        return False

    def _init_modality_writer(self, modality_metadata: BaseModalityMetadata) -> None:
        """Create the Arrow writer(s) for a single modality metadata entry."""
        assert self._state is not None, "Log writer state is not initialized."
        if isinstance(modality_metadata, EgoStateSE3Metadata):
            if self._dataset_converter_config.include_ego:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowEgoStateSE3Writer(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, BoxDetectionsSE3Metadata):
            if self._dataset_converter_config.include_box_detections:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowBoxDetectionsSE3Writer(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, TrafficLightDetectionsMetadata):
            if self._dataset_converter_config.include_traffic_lights:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowTrafficLightDetectionsWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, PinholeCameraMetadata):
            if self._dataset_converter_config.include_pinhole_cameras:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowPinholeCameraWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    data_codec=self._dataset_converter_config.pinhole_camera_store_option,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, FisheyeMEICameraMetadata):
            if self._dataset_converter_config.include_fisheye_mei_cameras:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowFisheyeMEICameraWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    data_codec=self._dataset_converter_config.fisheye_mei_camera_store_option,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, (LidarMergedMetadata, LidarMetadata)):
            if self._dataset_converter_config.include_lidars:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowLidarWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    log_metadata=self._state.log_metadata,
                    lidar_store_option=self._dataset_converter_config.lidar_store_option,
                    lidar_point_cloud_codec=self._dataset_converter_config.lidar_point_cloud_codec,
                    lidar_point_feature_codec=self._dataset_converter_config.lidar_point_feature_codec,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, CustomModalityMetadata):
            self._state.modality_writers[modality_metadata.modality_name] = ArrowCustomModalityWriter(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )

        else:
            raise ValueError(f"Unsupported modality metadata type: {type(modality_metadata)}")

    # ------------------------------------------------------------------------------------------------------------------
    # Writing: unified dispatch via FrameData.iter_modalities()
    # ------------------------------------------------------------------------------------------------------------------

    def _write_modalities(self, frame: FrameData, track_timestamps: bool = False) -> Dict[str, List[int]]:
        """Dispatch each modality in *frame* to its writer.

        :param frame: The frame data to write.
        :param track_timestamps: If True, record (row_index, timestamp_us) for deferred sync.
        :return: Dict mapping modality_name -> [row_index] for sync-table construction (sync mode only).
        """
        assert self._state is not None
        sync_row_indices: Dict[str, List[int]] = {}

        for modality_name, data in frame.iter_modalities():
            writer = self._state.modality_writers.get(modality_name)
            if writer is None:
                # Writer not registered (modality filtered out by config, or custom modality not yet seen)
                if isinstance(data, self._LAZY_INIT_TYPES):
                    self._init_modality_writer(data.metadata)
                    writer = self._state.modality_writers.get(modality_name)
                if writer is None:
                    continue

            row_idx = writer.row_count
            writer.write_modality(data)

            # For sync mode: capture the row index per modality
            sync_row_indices[modality_name] = [row_idx]

            # For async/deferred sync mode: buffer (row_index, timestamp_us)
            if track_timestamps:
                ts_us = _get_timestamp_us_from_data(data, frame.timestamp)
                self._state.timestamp_log[modality_name].append((row_idx, ts_us))

        return sync_row_indices

    # Modality data types that support lazy writer initialization (have a .metadata attribute)
    _LAZY_INIT_TYPES: tuple = ()  # Populated after imports; currently only CustomModality

    def write_sync(self, frame: FrameData) -> None:
        """Write one synchronized frame — all modalities plus one sync-table row."""
        assert self._state is not None, "Log writer is not initialized. Call reset() first."

        sync_row_indices = self._write_modalities(frame)

        # Build the sync row
        frame_uuid = frame.uuid
        if frame_uuid is None:
            frame_uuid = create_deterministic_uuid(
                split=self._state.log_metadata.split,
                log_name=self._state.log_metadata.log_name,
                timestamp_us=frame.timestamp.time_us,
            )

        sync_writer = self._state.modality_writers.get("sync")
        if sync_writer is None:
            # Create the sync writer on first write, now that we know which modalities are present
            self._create_sync_writer(list(sync_row_indices.keys()))
            sync_writer = self._state.modality_writers["sync"]

        sync_data: Dict[str, Any] = {
            SYNC.col("uuid"): [frame_uuid.bytes],
            SYNC.col("timestamp_us"): [frame.timestamp.time_us],
        }
        for modality_name, row_indices in sync_row_indices.items():
            sync_data[modality_name] = [row_indices]
        sync_writer.write_batch(sync_data)

    def write_async(self, frame: FrameData, modality_name: str) -> None:
        """Write a single async modality observation from *frame*."""
        assert self._state is not None, "Log writer is not initialized. Call reset() first."
        self._write_modalities(frame, track_timestamps=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Sync table helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _create_sync_writer(self, addon_modality_names: List[str]) -> None:
        """Create the sync.arrow writer with columns for uuid, timestamp, plus one list column per modality."""
        assert self._state is not None
        schema_fields: List[Tuple[str, pa.DataType]] = [
            (SYNC.col("uuid"), SYNC.columns["uuid"]),
            (SYNC.col("timestamp_us"), SYNC.columns["timestamp_us"]),
        ]
        for name in addon_modality_names:
            schema_fields.append((name, pa.list_(pa.int64())))

        schema = pa.schema(schema_fields)
        schema = add_metadata_to_arrow_schema(schema, self._state.log_metadata)

        sync_writer = BaseModalityWriter(
            file_path=self._state.log_dir / "sync.arrow",
            schema=schema,
            ipc_compression=self._ipc_compression,
            ipc_compression_level=self._ipc_compression_level,
        )
        self._state.modality_writers["sync"] = sync_writer

    def _build_deferred_sync_table(self) -> None:
        """Build the sync table from buffered timestamps using lidar sweep intervals.

        Each sync row corresponds to one lidar sweep. The lidar reference timestamp is the
        **end** of the sweep (matching the convention that annotations and ego poses are
        timestamped at end-of-sweep). The interval for sweep *i* is
        ``[lidar_end_ts_i, lidar_end_ts_{i+1})``. For each modality addon, the row
        contains the list of modality row indices whose timestamps fall within that interval.
        """
        assert self._state is not None

        # Find the lidar modality used as the sync reference
        lidar_modality_name = f"lidar.{LidarID.LIDAR_MERGED.serialize()}"
        lidar_entries = self._state.timestamp_log.get(lidar_modality_name, [])
        if not lidar_entries:
            return

        # Sort lidar entries by timestamp
        lidar_entries.sort(key=lambda e: e[1])
        lidar_timestamps_us = [ts for _, ts in lidar_entries]

        # Addon modality names: everything including the lidar reference (the API expects it in sync)
        addon_names = list(self._state.timestamp_log.keys())

        # Create sync writer
        self._create_sync_writer(addon_names)
        sync_writer = self._state.modality_writers["sync"]

        # Pre-sort each addon's timestamp log for efficient interval lookup
        sorted_logs: Dict[str, List[Tuple[int, int]]] = {}
        for addon in addon_names:
            entries = self._state.timestamp_log.get(addon, [])
            sorted_logs[addon] = sorted(entries, key=lambda e: e[1])

        # Build one sync row per lidar sweep
        for sweep_idx, (_, lidar_ts) in enumerate(lidar_entries):
            next_ts = lidar_timestamps_us[sweep_idx + 1] if sweep_idx + 1 < len(lidar_entries) else None

            sync_addon_data: Dict[str, List[int]] = {}
            for addon in addon_names:
                addon_entries = sorted_logs[addon]
                addon_timestamps = [ts for _, ts in addon_entries]

                # Find row indices in [lidar_ts, next_ts)
                lo = bisect.bisect_left(addon_timestamps, lidar_ts)
                hi = bisect.bisect_left(addon_timestamps, next_ts) if next_ts is not None else len(addon_timestamps)
                sync_addon_data[addon] = [addon_entries[i][0] for i in range(lo, hi)]

            sync_uuid = create_deterministic_uuid(
                split=self._state.log_metadata.split,
                log_name=self._state.log_metadata.log_name,
                timestamp_us=lidar_ts,
            )

            sync_data: Dict[str, Any] = {
                SYNC.col("uuid"): [sync_uuid.bytes],
                SYNC.col("timestamp_us"): [lidar_ts],
            }
            for addon, row_indices in sync_addon_data.items():
                sync_data[addon] = [row_indices]
            sync_writer.write_batch(sync_data)

    # ------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def close(self) -> None:
        """Inherited, see superclass."""
        if self._state is not None:
            if self._state.deferred_sync:
                self._build_deferred_sync_table()
            self._close_writers()

        self._state = None


def _get_timestamp_us_from_data(data: Any, fallback_timestamp: Timestamp) -> int:
    """Extract timestamp in microseconds from a modality data object.

    For lidar data, uses end_timestamp (sweep reference). For objects with a .timestamp
    attribute, uses that. Otherwise falls back to the frame timestamp.
    """
    # LidarData: use end_timestamp as the sync reference
    if hasattr(data, "end_timestamp"):
        return data.end_timestamp.time_us
    if hasattr(data, "timestamp") and data.timestamp is not None:
        return data.timestamp.time_us
    return fallback_timestamp.time_us
