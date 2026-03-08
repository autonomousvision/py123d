from functools import lru_cache
from pathlib import Path
from typing import Dict, Final, List, Optional, Union

import pyarrow as pa

from py123d.api.map.arrow.arrow_map_api import get_lru_cached_map_api
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.utils.arrow_getters import (
    get_box_detections_se3_from_arrow_table,
    get_camera_from_arrow_table,
    get_ego_state_se3_from_arrow_table,
    get_lidar_from_arrow_table,
    get_timestamp_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from py123d.api.scene.arrow.utils.arrow_metadata_utils import get_metadata_from_arrow_schema
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table, open_arrow_schema
from py123d.api.utils.arrow_schema import (
    BOX_DETECTIONS_SE3,
    CUSTOM_MODALITY,
    EGO_STATE_SE3,
    FISHEYE_MEI_CAMERA,
    LIDAR,
    PINHOLE_CAMERA,
    SYNC,
    TRAFFIC_LIGHT_DETECTIONS,
)
from py123d.common.dataset_paths import get_dataset_paths
from py123d.common.utils.msgpack_utils import msgpack_decode_with_numpy
from py123d.common.utils.uuid_utils import convert_to_str_uuid
from py123d.datatypes import (
    BoxDetectionMetadata,
    BoxDetectionsSE3,
    CustomModality,
    EgoMetadata,
    EgoStateSE3,
    FisheyeMEICamera,
    FisheyeMEICameraID,
    FisheyeMEICameraMetadata,
    Lidar,
    LidarID,
    LidarMetadata,
    LogMetadata,
    MapMetadata,
    PinholeCamera,
    PinholeCameraID,
    PinholeCameraMetadata,
    Timestamp,
    TrafficLightDetections,
)
from py123d.datatypes.sensors.lidar import LidarMergedMetadata

# TODO: Refactor
MAX_LRU_CACHED_LOG_METADATA: Final[int] = 1_000


def _get_complete_log_scene_metadata(log_dir: Union[Path, str], log_metadata: LogMetadata) -> SceneMetadata:
    """Helper function to get the scene metadata for a complete log from a log directory."""
    sync_path = Path(log_dir) / f"{SYNC.prefix()}.arrow"
    table = get_lru_cached_arrow_table(sync_path)
    initial_uuid = convert_to_str_uuid(table[SYNC.col("uuid")][0].as_py())
    num_rows = table.num_rows
    return SceneMetadata(
        initial_uuid=initial_uuid,
        initial_idx=0,
        duration_s=log_metadata.timestep_seconds * num_rows,
        history_s=0.0,
        iteration_duration_s=log_metadata.timestep_seconds,
    )


@lru_cache(maxsize=MAX_LRU_CACHED_LOG_METADATA)
def _get_lru_cached_log_metadata(log_dir: Union[Path, str]) -> LogMetadata:
    """Helper function to get the log metadata for a log directory."""
    sync_schema = open_arrow_schema(Path(log_dir) / f"{SYNC.prefix()}.arrow")
    return get_metadata_from_arrow_schema(sync_schema, LogMetadata)


class ArrowSceneAPI(SceneAPI):
    """Scene API for Arrow-based scenes. Loads each modality from a separate Arrow file in a log directory."""

    def __init__(
        self,
        log_dir: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
    ) -> None:
        """Initializes the :class:`ArrowSceneAPI`.

        :param log_dir: Path to the log directory containing per-modality Arrow files.
        :param scene_metadata: Scene metadata, defaults to None
        """

        self._log_dir: Path = Path(log_dir)
        self._scene_metadata: Optional[SceneMetadata] = scene_metadata

        # Cache reference path for map API.
        self._map_file: Optional[Path] = None

    # Helper methods
    # ------------------------------------------------------------------------------------------------------------------

    def __reduce__(self):
        """Helper for pickling the object."""
        return (
            self.__class__,
            (self._log_dir, self._scene_metadata),
        )

    def _get_modality_table(self, modality_name: str) -> Optional[pa.Table]:
        """Load the Arrow table for the given modality, or None if the file does not exist."""
        table: Optional[pa.Table] = None
        file_path = self._log_dir / f"{modality_name}.arrow"
        if file_path.exists():
            table = get_lru_cached_arrow_table(file_path)
        return table

    def _get_sync_table(self) -> pa.Table:
        """Load the sync table. This must always exist."""
        table = self._get_modality_table(SYNC.prefix())
        assert table is not None, f"sync.arrow not found in {self._log_dir}"
        return table

    def _get_table_index(self, iteration: int) -> int:
        """Helper function to get the table index for a given iteration."""
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        table_index = self.get_scene_metadata().initial_idx + iteration
        return table_index

    @staticmethod
    def _get_first_sync_index(sync_table: pa.Table, column_name: str, idx: int) -> Optional[int]:
        """Extracts the first row index from a sync table column.

        Handles both scalar (pa.int64) and list-typed (pa.list_(pa.int64())) sync columns.
        For list-typed columns, returns the first element of the list (earliest observation in the interval).
        """
        value = sync_table[column_name][idx].as_py()
        if value is None:
            return None
        if isinstance(value, list):
            return value[0] if len(value) > 0 else None
        return value

    def _get_map_file(self) -> Optional[Path]:
        # FIXME: THis is hacky af.
        if self._map_file is None:
            # 1. Case: Map per log
            map_file = self._log_dir / "map.arrow"
            if map_file.exists():
                # 1. Case: Map per log
                self._map_file = map_file
            else:
                # 2. Case: Global map
                log_metadata = self.get_log_metadata()
                dataset, location = log_metadata.dataset, log_metadata.location
                if dataset is not None and location is not None:
                    _maps_root = get_dataset_paths().py123d_maps_root
                    if _maps_root is not None:
                        map_file = _maps_root / dataset / f"{dataset}_{location}.arrow"
                        if map_file.exists():
                            self._map_file = map_file
        return self._map_file

    # ------------------------------------------------------------------------------------------------------------------
    # Per-modality metadata retrieval (read from the corresponding Arrow schema)
    # ------------------------------------------------------------------------------------------------------------------

    def get_ego_state_se3_metadata(self) -> Optional[EgoMetadata]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoMetadata` read from ``ego_state_se3.arrow``."""
        ego_metadata: Optional[EgoMetadata] = None
        ego_table = self._get_modality_table(EGO_STATE_SE3.prefix())
        if ego_table is not None:
            ego_metadata = get_metadata_from_arrow_schema(ego_table.schema, EgoMetadata)
        return ego_metadata

    def get_box_detections_se3_metadata(self) -> Optional[BoxDetectionMetadata]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionMetadata` from ``box_detections_se3.arrow``."""
        box_detection_metadata: Optional[BoxDetectionMetadata] = None
        box_table = self._get_modality_table(BOX_DETECTIONS_SE3.prefix())
        if box_table is not None:
            box_detection_metadata = get_metadata_from_arrow_schema(box_table.schema, BoxDetectionMetadata)
        return box_detection_metadata

    def get_pinhole_camera_metadatas(self) -> Optional[Dict[PinholeCameraID, PinholeCameraMetadata]]:
        """Returns per-camera :class:`~py123d.datatypes.sensors.PinholeCameraMetadata` from the Arrow schema.

        Discovers ``pinhole_camera.{instance}.arrow`` files in the log directory and reads
        :class:`PinholeCameraMetadata` from each file's Arrow schema metadata.
        """
        result: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
        for arrow_file in sorted(self._log_dir.glob("pinhole_camera.*.arrow")):
            instance = arrow_file.stem.split(".", 1)[1]  # e.g. "pcam_f0" from "pinhole_camera.pcam_f0"
            camera_id: PinholeCameraID = PinholeCameraID.deserialize(instance)  # type: ignore[assignment]
            schema = open_arrow_schema(arrow_file)
            cam_meta = get_metadata_from_arrow_schema(schema, PinholeCameraMetadata)
            result[camera_id] = cam_meta
        return result if result else None

    def get_fisheye_mei_camera_metadatas(self) -> Optional[Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]]:
        """Returns per-camera :class:`~py123d.datatypes.sensors.FisheyeMEICameraMetadata` from the Arrow schema.

        Discovers ``fisheye_mei_camera.{instance}.arrow`` files in the log directory.
        """
        result: Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata] = {}
        for arrow_file in sorted(self._log_dir.glob("fisheye_mei_camera.*.arrow")):
            instance = arrow_file.stem.split(".", 1)[1]
            camera_id: FisheyeMEICameraID = FisheyeMEICameraID.deserialize(instance)  # type: ignore[assignment]
            schema = open_arrow_schema(arrow_file)
            cam_meta = get_metadata_from_arrow_schema(schema, FisheyeMEICameraMetadata)
            result[camera_id] = cam_meta
        return result if result else None

    def get_lidar_metadatas(self) -> Optional[Dict[LidarID, LidarMetadata]]:
        """Returns per-lidar :class:`~py123d.datatypes.sensors.LidarMetadata` from the lidar Arrow schema.

        Discovers ``lidar.{instance}.arrow`` files in the log directory.
        """
        result: Dict[LidarID, LidarMetadata] = {}
        for arrow_file in sorted(self._log_dir.glob("lidar.*.arrow")):
            schema = open_arrow_schema(arrow_file)
            if "lidar_merged" in arrow_file.stem:
                lidar_merged_meta = get_metadata_from_arrow_schema(schema, LidarMergedMetadata)
                result.update(lidar_merged_meta._data)  # type: ignore[union-attr]
            else:
                instance = arrow_file.stem.split(".", 1)[1]
                lidar_id: LidarID = LidarID.deserialize(instance)  # type: ignore[assignment]
                lidar_meta = get_metadata_from_arrow_schema(schema, LidarMetadata)
                result[lidar_id] = lidar_meta
        return result if result else None

    # Implementation of abstract methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return _get_lru_cached_log_metadata(self._log_dir)

    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see superclass."""
        if self._scene_metadata is None:
            log_metadata = self.get_log_metadata()
            self._scene_metadata = _get_complete_log_scene_metadata(self._log_dir, log_metadata)
        return self._scene_metadata

    def get_map_metadata(self) -> Optional[MapMetadata]:
        """Inherited, see superclass."""

        _map_file = self._get_map_file()
        if _map_file is not None:
            map_schema = open_arrow_schema(_map_file)
            map_metadata = get_metadata_from_arrow_schema(map_schema, MapMetadata)
        else:
            map_metadata: Optional[MapMetadata] = None

        return map_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see superclass."""
        map_api: Optional[MapAPI] = None
        _map_file = self._get_map_file()
        if _map_file is not None:
            map_api = get_lru_cached_map_api(_map_file)
        return map_api

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Inherited, see superclass."""
        return get_timestamp_from_arrow_table(self._get_sync_table(), self._get_table_index(iteration))

    def get_ego_state_se3_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Inherited, see superclass."""
        ego_state: Optional[EgoStateSE3] = None
        ego_table = self._get_modality_table(EGO_STATE_SE3.prefix())
        if ego_table is not None:
            sync_table = self._get_sync_table()
            idx = self._get_table_index(iteration)
            row_idx = self._get_first_sync_index(sync_table, EGO_STATE_SE3.prefix(), idx)
            if row_idx is not None:
                ego_metadata = get_metadata_from_arrow_schema(ego_table.schema, EgoMetadata)
                ego_state = get_ego_state_se3_from_arrow_table(ego_table, row_idx, ego_metadata)
        return ego_state

    def get_box_detections_se3_at_iteration(self, iteration: int) -> Optional[BoxDetectionsSE3]:
        """Inherited, see superclass."""
        box_detections: Optional[BoxDetectionsSE3] = None
        box_table = self._get_modality_table(BOX_DETECTIONS_SE3.prefix())
        if box_table is not None:
            sync_table = self._get_sync_table()
            idx = self._get_table_index(iteration)
            row_idx = self._get_first_sync_index(sync_table, BOX_DETECTIONS_SE3.prefix(), idx)
            if row_idx is not None:
                box_detection_metadata = get_metadata_from_arrow_schema(box_table.schema, BoxDetectionMetadata)
                if box_detection_metadata is not None:
                    timestamp = self.get_timestamp_at_iteration(iteration)
                    box_detections = get_box_detections_se3_from_arrow_table(
                        box_table, row_idx, box_detection_metadata, timestamp
                    )
        return box_detections

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetections]:
        """Inherited, see superclass."""
        traffic_light_detections: Optional[TrafficLightDetections] = None
        tl_table = self._get_modality_table(TRAFFIC_LIGHT_DETECTIONS.prefix())
        if tl_table is not None:
            sync_table = self._get_sync_table()
            idx = self._get_table_index(iteration)
            row_idx = self._get_first_sync_index(sync_table, TRAFFIC_LIGHT_DETECTIONS.prefix(), idx)
            if row_idx is not None:
                traffic_light_detections = get_traffic_light_detections_from_arrow_table(tl_table, row_idx)
        return traffic_light_detections

    def get_pinhole_camera_at_iteration(self, iteration: int, camera_id: PinholeCameraID) -> Optional[PinholeCamera]:
        """Inherited, see superclass."""
        pinhole_camera: Optional[PinholeCamera] = None
        camera_instance = camera_id.serialize()
        modality_name = PINHOLE_CAMERA.prefix(camera_instance)
        cam_table = self._get_modality_table(modality_name)
        if cam_table is not None:
            cam_meta = get_metadata_from_arrow_schema(cam_table.schema, PinholeCameraMetadata)
            sync_table = self._get_sync_table()
            idx = self._get_table_index(iteration)
            row_idx = self._get_first_sync_index(sync_table, modality_name, idx)
            if row_idx is not None:
                pinhole_camera = get_camera_from_arrow_table(cam_table, row_idx, camera_id, cam_meta, self.log_metadata)  # type: ignore[return-value]
        return pinhole_camera

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_id: FisheyeMEICameraID
    ) -> Optional[FisheyeMEICamera]:
        """Inherited, see superclass."""
        fisheye_mei_camera: Optional[FisheyeMEICamera] = None
        camera_instance = camera_id.serialize()
        modality_name = FISHEYE_MEI_CAMERA.prefix(camera_instance)
        cam_table = self._get_modality_table(modality_name)
        if cam_table is not None:
            cam_meta = get_metadata_from_arrow_schema(cam_table.schema, FisheyeMEICameraMetadata)
            sync_table = self._get_sync_table()
            idx = self._get_table_index(iteration)
            row_idx = self._get_first_sync_index(sync_table, modality_name, idx)
            if row_idx is not None:
                fisheye_mei_camera = get_camera_from_arrow_table(
                    cam_table, row_idx, camera_id, cam_meta, self.log_metadata
                )  # type: ignore[return-value]
        return fisheye_mei_camera

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> Optional[Lidar]:
        """Inherited, see superclass."""
        lidar: Optional[Lidar] = None
        lidar_instance = LidarID.LIDAR_MERGED.serialize()
        modality_name = LIDAR.prefix(lidar_instance)
        lidar_table = self._get_modality_table(modality_name)
        if lidar_table is not None:
            lidar_metadatas = self.get_lidar_metadatas()
            if lidar_metadatas is not None and lidar_id in list(lidar_metadatas.keys()) + [LidarID.LIDAR_MERGED]:
                sync_table = self._get_sync_table()
                idx = self._get_table_index(iteration)
                row_idx = self._get_first_sync_index(sync_table, modality_name, idx)
                if row_idx is not None:
                    lidar = get_lidar_from_arrow_table(
                        lidar_table, row_idx, lidar_id, lidar_instance, lidar_metadatas, self.log_metadata
                    )
        return lidar

    def get_ego_states_se3_in_window(self, start_timestamp: Timestamp, end_timestamp: Timestamp) -> List[EgoStateSE3]:
        """Returns all ego states with timestamps in [start_timestamp, end_timestamp).

        Reads the ego state table directly by timestamp, independent of the sync table.

        :param start_timestamp: Inclusive start of the window.
        :param end_timestamp: Exclusive end of the window.
        :return: List of ego states within the window, sorted by timestamp.
        """
        ego_table = self._get_modality_table(EGO_STATE_SE3.prefix())
        if ego_table is None:
            return []

        ego_metadata = get_metadata_from_arrow_schema(ego_table.schema, EgoMetadata)
        ts_column = ego_table[EGO_STATE_SE3.col("timestamp_us")]

        result: List[EgoStateSE3] = []
        for row_idx in range(ego_table.num_rows):
            ts_us = ts_column[row_idx].as_py()
            if start_timestamp.time_us <= ts_us < end_timestamp.time_us:
                ego_state = get_ego_state_se3_from_arrow_table(ego_table, row_idx, ego_metadata)
                if ego_state is not None:
                    result.append(ego_state)

        return result

    def get_custom_modality_at_iteration(self, iteration: int, name: str) -> Optional[CustomModality]:
        """Inherited, see superclass."""
        custom_modality: Optional[CustomModality] = None
        table = self._get_modality_table(CUSTOM_MODALITY.prefix(name))
        if table is not None:
            idx = self._get_table_index(iteration)
            encoded_data: bytes = table[CUSTOM_MODALITY.col("data", name)][idx].as_py()
            timestamp_us: int = table[CUSTOM_MODALITY.col("timestamp_us", name)][idx].as_py()
            data = msgpack_decode_with_numpy(encoded_data)
            custom_modality = CustomModality(data=data, timestamp=Timestamp.from_us(timestamp_us))
        return custom_modality
