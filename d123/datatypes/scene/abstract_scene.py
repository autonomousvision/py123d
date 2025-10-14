from __future__ import annotations

import abc
from typing import List, Optional, Union

from d123.datatypes.detections.detection import BoxDetectionWrapper, DetectionRecording, TrafficLightDetectionWrapper
from d123.datatypes.maps.abstract_map import AbstractMap
from d123.datatypes.scene.scene_metadata import LogMetadata, SceneExtractionMetadata
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCamera, PinholeCameraType
from d123.datatypes.sensors.camera.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraType
from d123.datatypes.sensors.lidar.lidar import LiDAR, LiDARType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import EgoStateSE3
from d123.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


class AbstractScene(abc.ABC):

    ####################################################################################################################
    # Abstract Methods, to be implemented by subclasses
    ####################################################################################################################

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scene_extraction_metadata(self) -> SceneExtractionMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def get_map_api(self) -> Optional[AbstractMap]:
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
    def get_camera_at_iteration(self, iteration: int, camera_type: Union[PinholeCameraType, FisheyeMEICameraType]) -> Optional[Union[PinholeCamera, FisheyeMEICamera]]:
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
    def available_camera_types(self) -> List[Union[PinholeCameraType, FisheyeMEICameraType]]:
        return list(self.log_metadata.camera_metadata.keys())

    @property
    def available_lidar_types(self) -> List[LiDARType]:
        return list(self.log_metadata.lidar_metadata.keys())

    # 2. Scene Extraction Metadata properties
    @property
    def scene_extraction_metadata(self) -> SceneExtractionMetadata:
        return self.get_scene_extraction_metadata()

    @property
    def token(self) -> str:
        return self.scene_extraction_metadata.initial_token

    @property
    def number_of_iterations(self) -> int:
        return self.scene_extraction_metadata.number_of_iterations

    @property
    def number_of_history_iterations(self) -> int:
        return self.scene_extraction_metadata.number_of_history_iterations
