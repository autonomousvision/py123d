from __future__ import annotations

import abc

from asim.common.vehicle_state.ego_state import EgoVehicleState
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.observation.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper


class AbstractScene(abc.ABC):
    @property
    @abc.abstractmethod
    def map_api(self) -> AbstractMap:
        raise NotImplementedError

    @abc.abstractmethod
    def get_ego_vehicle_state_at_iteration(self, iteration: int) -> EgoVehicleState:
        raise NotImplementedError

    @abc.abstractmethod
    def get_box_detections_at_iteration(self, iteration: int) -> BoxDetectionWrapper:
        raise NotImplementedError

    @abc.abstractmethod
    def get_traffic_light_detections_at_iteration(self, iteration: int) -> TrafficLightDetectionWrapper:
        raise NotImplementedError
