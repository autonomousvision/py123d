from __future__ import annotations

import abc
from typing import List, Optional

from py123d.api.map.map_api import MapAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDAR, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


class SceneAPI(abc.ABC):

    ####################################################################################################################
    # Abstract Methods, to be implemented by subclasses
    ####################################################################################################################

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scene_extraction_metadata(self) -> SceneMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def get_map_api(self) -> Optional[MapAPI]:
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
    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_pinhole_camera_at_iteration(
        self, iteration: int, camera_type: PinholeCameraType
    ) -> Optional[PinholeCamera]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_type: FisheyeMEICameraType
    ) -> Optional[FisheyeMEICamera]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_lidar_at_iteration(self, iteration: int, lidar_type: LiDARType) -> Optional[LiDAR]:
        raise NotImplementedError

    ####################################################################################################################
    # Syntactic Sugar / Properties, for easier access to common attributes
    ####################################################################################################################

    # 1. Log Metadata properties
    @property
    def log_metadata(self) -> LogMetadata:
        return self.get_log_metadata()

    @property
    def log_name(self) -> str:
        return self.log_metadata.log_name

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        return self.log_metadata.vehicle_parameters

    @property
    def available_pinhole_camera_types(self) -> List[PinholeCameraType]:
        return list(self.log_metadata.pinhole_camera_metadata.keys())

    @property
    def available_fisheye_mei_camera_types(self) -> List[FisheyeMEICameraType]:
        return list(self.log_metadata.fisheye_mei_camera_metadata.keys())

    @property
    def available_lidar_types(self) -> List[LiDARType]:
        return list(self.log_metadata.lidar_metadata.keys())

    # 2. Scene Extraction Metadata properties
    @property
    def scene_extraction_metadata(self) -> SceneMetadata:
        return self.get_scene_extraction_metadata()

    @property
    def uuid(self) -> str:
        return self.scene_extraction_metadata.initial_uuid

    @property
    def number_of_iterations(self) -> int:
        return self.scene_extraction_metadata.number_of_iterations

    @property
    def number_of_history_iterations(self) -> int:
        return self.scene_extraction_metadata.number_of_history_iterations
