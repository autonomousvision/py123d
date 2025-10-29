from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.camera.pinhole_camera import PinholeCameraType
from py123d.datatypes.sensors.lidar.lidar import LiDARType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3


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
        timestamp: TimePoint,
        ego_state: Optional[EgoStateSE3] = None,
        box_detections: Optional[BoxDetectionWrapper] = None,
        traffic_lights: Optional[TrafficLightDetectionWrapper] = None,
        cameras: Optional[Dict[PinholeCameraType, Tuple[Any, ...]]] = None,
        lidars: Optional[List[LiDARData]] = None,
        scenario_tags: Optional[List[str]] = None,
        route_lane_group_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


@dataclass
class LiDARData:

    lidar_type: LiDARType

    timestamp: Optional[TimePoint] = None
    iteration: Optional[int] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        has_file_path = self.dataset_root is not None and self.relative_path is not None

        assert has_file_path, "Either file path (dataset_root and relative_path) must be provided for LiDARData."


@dataclass
class CameraData:

    camera_type: PinholeCameraType

    timestamp: Optional[TimePoint] = None
    jpeg_binary: Optional[bytes] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        has_file_path = self.dataset_root is not None and self.relative_path is not None
        has_jpeg_binary = self.jpeg_binary is not None

        assert (
            has_file_path or has_jpeg_binary
        ), "Either file path (dataset_root and relative_path) or jpeg_binary must be provided for CameraData."
