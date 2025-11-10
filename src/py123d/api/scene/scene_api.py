from __future__ import annotations

import abc
from typing import List, Optional

from py123d.api.map.map_api import MapAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDAR, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


class SceneAPI(abc.ABC):

    # Abstract Methods, to be implemented by subclasses
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        """Returns the :class:`~py123d.datatypes.metadata.LogMetadata` of the scene.

        :return: The log metadata.
        """

    @abc.abstractmethod
    def get_scene_metadata(self) -> SceneMetadata:
        """Returns the :class:`~py123d.api.scene.scene_metadata.SceneMetadata` of the scene.

        :return: The scene metadata.
        """

    @abc.abstractmethod
    def get_map_api(self) -> Optional[MapAPI]:
        """Returns the :class:`~py123d.api.MapAPI` of the scene, if available.

        :return: The map API, or None if not available.
        """

    @abc.abstractmethod
    def get_timepoint_at_iteration(self, iteration: int) -> TimePoint:
        """Returns the :class:`~py123d.datatypes.time.TimePoint` at a given iteration.

        :param iteration: The iteration to get the timepoint for.
        :return: The timepoint at the given iteration.
        """

    @abc.abstractmethod
    def get_ego_state_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Returns the :class:`~py123d.datatypes.vehicle_state.EgoStateSE3` at a given iteration, if available.

        :param iteration: The iteration to get the ego state for.
        :return: The ego state at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionWrapper]:
        """Returns the :class:`~py123d.datatypes.detections.BoxDetectionWrapper` at a given iteration, if available.

        :param iteration: The iteration to get the box detections for.
        :return: The box detections at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetectionWrapper]:
        """Returns the :class:`~py123d.datatypes.detections.TrafficLightDetectionWrapper` at a given iteration,
            if available.

        :param iteration: The iteration to get the traffic light detections for.
        :return: The traffic light detections at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        """Returns the list of route lane group IDs at a given iteration, if available.

        :param iteration: The iteration to get the route lane group IDs for.
        :return: The list of route lane group IDs at the given iteration, or None if not available.
        """

    @abc.abstractmethod
    def get_pinhole_camera_at_iteration(
        self,
        iteration: int,
        camera_type: PinholeCameraType,
    ) -> Optional[PinholeCamera]:
        """Returns the :class:`~py123d.datatypes.sensors.PinholeCamera` of a given \
            :class:`~py123d.datatypes.sensors.PinholeCameraType` at a given iteration, if available.

        :param iteration: The iteration to get the pinhole camera for.
        :param camera_type: The :type:`~py123d.datatypes.sensors.PinholeCameraType` of the pinhole camera.
        :return: The pinhole camera, or None if not available.
        """

    @abc.abstractmethod
    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_type: FisheyeMEICameraType
    ) -> Optional[FisheyeMEICamera]:
        """Returns the :class:`~py123d.datatypes.sensors.FisheyeMEICamera` of a given \
            :class:`~py123d.datatypes.sensors.FisheyeMEICameraType` at a given iteration, if available.

        :param iteration: The iteration to get the fisheye MEI camera for.
        :param camera_type: The :type:`~py123d.datatypes.sensors.FisheyeMEICameraType` of the fisheye MEI camera.
        :return: The fisheye MEI camera, or None if not available.
        """

    @abc.abstractmethod
    def get_lidar_at_iteration(self, iteration: int, lidar_type: LiDARType) -> Optional[LiDAR]:
        """Returns the :class:`~py123d.datatypes.sensors.LiDAR` of a given :class:`~py123d.datatypes.sensors.LiDARType`\
            at a given iteration, if available.

        :param iteration: The iteration to get the LiDAR for.
        :param lidar_type: The :type:`~py123d.datatypes.sensors.LiDARType` of the LiDAR.
        :return: The LiDAR, or None if not available.
        """

    # Syntactic Sugar / Properties, for easier access to common attributes
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def log_metadata(self) -> LogMetadata:
        """The :class:`~py123d.datatypes.metadata.LogMetadata` of the scene."""
        return self.get_log_metadata()

    @property
    def scene_metadata(self) -> SceneMetadata:
        """The :class:`~py123d.api.scene.SceneMetadata` of the scene."""
        return self.get_scene_metadata()

    @property
    def map_metadata(self) -> Optional[MapMetadata]:
        """The :class:`~py123d.datatypes.metadata.MapMetadata` of the scene, if available."""
        return self.log_metadata.map_metadata

    @property
    def map_api(self) -> Optional[MapAPI]:
        """The :class:`~py123d.api.map.MapAPI` of the scene, if available."""
        return self.get_map_api()

    @property
    def dataset(self) -> str:
        """The dataset name from the log metadata."""
        return self.log_metadata.dataset

    @property
    def split(self) -> str:
        """The data split name from the log metadata."""
        return self.log_metadata.split

    @property
    def location(self) -> str:
        """The location from the log metadata."""
        return self.log_metadata.location

    @property
    def log_name(self) -> str:
        """The log name from the log metadata."""
        return self.log_metadata.log_name

    @property
    def version(self) -> str:
        """The version of the py123d library used to create this log metadata."""
        return self.log_metadata.version

    @property
    def scene_uuid(self) -> str:
        """The UUID of the scene."""
        return self.scene_metadata.initial_uuid

    @property
    def number_of_iterations(self) -> int:
        """The number of iterations in the scene."""
        return self.scene_metadata.number_of_iterations

    @property
    def number_of_history_iterations(self) -> int:
        """The number of history iterations in the scene."""
        return self.scene_metadata.number_of_history_iterations

    @property
    def vehicle_parameters(self) -> Optional[VehicleParameters]:
        """The :class:`~py123d.datatypes.vehicle_state.VehicleParameters` of the ego vehicle, if available."""
        return self.log_metadata.vehicle_parameters

    @property
    def available_pinhole_camera_types(self) -> List[PinholeCameraType]:
        """List of available :class:`~py123d.datatypes.sensors.PinholeCameraType` in the log metadata."""
        return list(self.log_metadata.pinhole_camera_metadata.keys())

    @property
    def available_fisheye_mei_camera_types(self) -> List[FisheyeMEICameraType]:
        """List of available :class:`~py123d.datatypes.sensors.FisheyeMEICameraType` in the log metadata."""
        return list(self.log_metadata.fisheye_mei_camera_metadata.keys())

    @property
    def available_lidar_types(self) -> List[LiDARType]:
        """List of available :class:`~py123d.datatypes.sensors.LiDARType` in the log metadata."""
        return list(self.log_metadata.lidar_metadata.keys())
