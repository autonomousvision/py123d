from pathlib import Path
from typing import Optional

import pyarrow as pa

from asim.common.time.time_point import TimePoint
from asim.common.vehicle_state.ego_vehicle_state import EgoVehicleState
from asim.dataset.arrow.conversion import (
    get_box_detections_from_arrow_table,
    get_ego_vehicle_state_from_arrow_table,
    get_timepoint_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from asim.dataset.logs.log_metadata import LogMetadata
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.gpkg.gpkg_map import get_map_api_from_names
from asim.dataset.observation.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from asim.dataset.scene.abstract_scene import AbstractScene


class ArrowScene(AbstractScene):
    def __init__(self, arrow_file_path: Path) -> None:
        self._arrow_log_path = arrow_file_path

        with pa.memory_map(str(arrow_file_path), "rb") as source:
            recording_table = pa.ipc.open_file(source).read_all()

        self._recording_table: pa.Table = recording_table
        self._metadata: LogMetadata = LogMetadata.from_arrow_table(recording_table)
        self._map_api: Optional[AbstractMap] = None

    def __reduce__(self):
        return (self.__class__, (self._arrow_log_path,))

    @property
    def map_api(self) -> AbstractMap:
        self._lazy_initialize()
        return self._map_api

    @property
    def log_name(self) -> str:
        return str(self._arrow_log_path.stem)

    @property
    def log_metadata(self) -> LogMetadata:
        return self._metadata

    def get_number_of_iterations(self) -> int:
        self._lazy_initialize()
        return len(self._recording_table)

    def get_timepoint_at_iteration(self, iteration: int) -> TimePoint:
        self._lazy_initialize()
        return get_timepoint_from_arrow_table(self._recording_table, iteration)

    def get_ego_vehicle_state_at_iteration(self, iteration: int) -> EgoVehicleState:
        self._lazy_initialize()
        return get_ego_vehicle_state_from_arrow_table(self._recording_table, iteration)

    def get_box_detections_at_iteration(self, iteration: int) -> BoxDetectionWrapper:
        self._lazy_initialize()
        return get_box_detections_from_arrow_table(self._recording_table, iteration)

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> TrafficLightDetectionWrapper:
        self._lazy_initialize()
        return get_traffic_light_detections_from_arrow_table(self._recording_table, iteration)

    def _lazy_initialize(self) -> None:
        if self._map_api is None:
            self._map_api = get_map_api_from_names(self._metadata.dataset, self._metadata.location)
            self._map_api.initialize()
