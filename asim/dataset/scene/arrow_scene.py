from pathlib import Path

from asim.common.vehicle_state.ego_state import EgoVehicleState
from asim.dataset.arrow.multiple_table import ArrowMultiTableFile
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.gpkg.gpkg_map import get_map_api_from_names
from asim.dataset.observation.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from asim.dataset.scene.abstract_scene import AbstractScene


class ArrowScene(AbstractScene):
    def __init__(self, arrow_file_path: Path) -> None:

        self._arrow_multi_table = ArrowMultiTableFile(arrow_file_path)

        self._metadata_table = self._arrow_multi_table.get_table("metadata_table")
        self._recording_table = self._arrow_multi_table.get_table("recording_table")

        self._map_api: AbstractMap = get_map_api_from_names()

    @property
    def map_api(self) -> AbstractMap:
        return self._map_api

    def get_ego_vehicle_state_at_iteration(self, iteration: int) -> EgoVehicleState:
        raise NotImplementedError

    def get_box_detections_at_iteration(self, iteration: int) -> BoxDetectionWrapper:
        raise NotImplementedError

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> TrafficLightDetectionWrapper:
        raise NotImplementedError
