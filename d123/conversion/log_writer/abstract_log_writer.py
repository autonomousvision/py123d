import abc
from typing import Any, Dict, List, Optional, Tuple

from d123.conversion.dataset_converter_config import DatasetConverterConfig
from d123.datatypes.detections.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType
from d123.datatypes.sensors.lidar.lidar import LiDARType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import EgoStateSE3


class AbstractLogWriter(abc.ABC):
    """Abstract base class for log writers.

    A log writer is responsible specifying the output format of a converted log.
    This includes how data is organized, how it is serialized, and how it is stored.
    """

    @abc.abstractmethod
    def reset(
        self,
        dataset_converter_config: DatasetConverterConfig,
        log_metadata: LogMetadata,
    ) -> None:
        """
        Reset the log writer for a new log.
        """

    @abc.abstractmethod
    def write(
        self,
        token: str,
        timestamp: TimePoint,
        ego_state: Optional[EgoStateSE3] = None,
        box_detections: Optional[BoxDetectionWrapper] = None,
        traffic_lights: Optional[TrafficLightDetectionWrapper] = None,
        cameras: Optional[Dict[PinholeCameraType, Tuple[Any, ...]]] = None,
        lidars: Optional[Dict[LiDARType, Any]] = None,
        scenario_tags: Optional[List[str]] = None,
        route_lane_group_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass
