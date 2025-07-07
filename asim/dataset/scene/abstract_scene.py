from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List

from asim.common.datatypes.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from asim.common.datatypes.time.time_point import TimePoint
from asim.common.datatypes.vehicle_state.ego_state import EgoStateSE3
from asim.dataset.logs.log_metadata import LogMetadata
from asim.dataset.maps.abstract_map import AbstractMap


class AbstractScene(abc.ABC):
    @property
    @abc.abstractmethod
    def map_api(self) -> AbstractMap:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def log_name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def token(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def log_metadata(self) -> LogMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def get_number_of_iterations(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_number_of_history_iterations() -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_timepoint_at_iteration(self, iteration: int) -> TimePoint:
        raise NotImplementedError

    @abc.abstractmethod
    def get_ego_vehicle_state_at_iteration(self, iteration: int) -> EgoStateSE3:
        raise NotImplementedError

    @abc.abstractmethod
    def get_box_detections_at_iteration(self, iteration: int) -> BoxDetectionWrapper:
        raise NotImplementedError

    @abc.abstractmethod
    def get_traffic_light_detections_at_iteration(self, iteration: int) -> TrafficLightDetectionWrapper:
        raise NotImplementedError

    @abc.abstractmethod
    def get_route_lane_group_ids(self, iteration: int) -> List[int]:
        raise NotImplementedError

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class SceneExtractionInfo:

    initial_idx: int
    duration_s: float
    history_s: float
    iteration_duration_s: float

    @property
    def number_of_iterations(self) -> int:
        return round(self.duration_s / self.iteration_duration_s)

    @property
    def number_of_history_iterations(self) -> int:
        return round(self.history_s / self.iteration_duration_s)

    @property
    def end_idx(self) -> int:
        return self.initial_idx + self.number_of_iterations
