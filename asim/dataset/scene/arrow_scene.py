from pathlib import Path

from asim.common.time.time_point import TimePoint
from asim.common.vehicle_state.ego_vehicle_state import EgoVehicleState
from asim.dataset.arrow.conversion import (
    get_box_detections_from_arrow_table,
    get_ego_vehicle_state_from_arrow_table,
    get_timepoint_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from asim.dataset.arrow.multiple_table import ArrowMultiTableFile
from asim.dataset.logs.log_metadata import LogMetadata
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.gpkg.gpkg_map import get_map_api_from_names
from asim.dataset.observation.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from asim.dataset.scene.abstract_scene import AbstractScene


class ArrowScene(AbstractScene):
    def __init__(self, arrow_file_path: Path) -> None:

        self._arrow_multi_table = ArrowMultiTableFile(arrow_file_path)

        self._metadata: LogMetadata = LogMetadata.from_arrow_table(self._arrow_multi_table.get_table("metadata_table"))
        self._recording_table = self._arrow_multi_table.get_table("recording_table")

        self._map_api: AbstractMap = get_map_api_from_names(self._metadata.dataset, self._metadata.location)
        self._map_api.initialize()

    @property
    def map_api(self) -> AbstractMap:
        return self._map_api

    def get_timepoint_at_iteration(self, iteration: int) -> TimePoint:
        return get_timepoint_from_arrow_table(self._recording_table, iteration)

    def get_ego_vehicle_state_at_iteration(self, iteration: int) -> EgoVehicleState:
        return get_ego_vehicle_state_from_arrow_table(self._recording_table, iteration)

    def get_box_detections_at_iteration(self, iteration: int) -> BoxDetectionWrapper:
        return get_box_detections_from_arrow_table(self._recording_table, iteration)

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> TrafficLightDetectionWrapper:
        return get_traffic_light_detections_from_arrow_table(self._recording_table, iteration)
