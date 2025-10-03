from __future__ import annotations

import abc
from typing import List, Optional

from d123.datatypes.detections.detection import BoxDetectionWrapper, DetectionRecording, TrafficLightDetectionWrapper
from d123.datatypes.maps.abstract_map import AbstractMap
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.datatypes.sensors.camera import Camera, CameraType
from d123.datatypes.sensors.lidar import LiDAR, LiDARType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import EgoStateSE3

# TODO: Remove or improve open/close dynamic of Scene object.


class AbstractScene(abc.ABC):

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

    @property
    @abc.abstractmethod
    def available_camera_types(self) -> List[CameraType]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def available_lidar_types(self) -> List[LiDARType]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def map_api(self) -> Optional[AbstractMap]:
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
    def get_ego_state_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionWrapper]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetectionWrapper]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_detection_recording_at_iteration(self, iteration: int) -> Optional[DetectionRecording]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_camera_at_iteration(self, iteration: int, camera_type: CameraType) -> Optional[Camera]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_lidar_at_iteration(self, iteration: int, lidar_type: LiDARType) -> Optional[LiDAR]:
        raise NotImplementedError

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass
