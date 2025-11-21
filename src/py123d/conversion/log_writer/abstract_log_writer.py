from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.metadata import LogMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import PoseSE3


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
    ) -> bool:
        """Resets the log writer to start writing a new log according to the provided configuration and metadata.

        :param dataset_converter_config: The dataset converter configuration.
        :param log_metadata: The metadata for the log.
        :return: True if the current logs needs to be written, False otherwise.
        """

    @abc.abstractmethod
    def write(
        self,
        timestamp: TimePoint,
        ego_state: Optional[EgoStateSE3] = None,
        box_detections: Optional[BoxDetectionWrapper] = None,
        traffic_lights: Optional[TrafficLightDetectionWrapper] = None,
        pinhole_cameras: Optional[List[CameraData]] = None,
        fisheye_mei_cameras: Optional[List[CameraData]] = None,
        lidars: Optional[List[LiDARData]] = None,
        scenario_tags: Optional[List[str]] = None,
        route_lane_group_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Writes a single iteration of data to the log.

        :param timestamp: Required, the timestamp of the iteration.
        :param ego_state: Optional, the ego state of the vehicle, defaults to None.
        :param box_detections: Optional, the box detections, defaults to None
        :param traffic_lights: Optional, the traffic light detections, defaults to None
        :param pinhole_cameras: Optional, the pinhole camera data, defaults to None
        :param fisheye_mei_cameras: Optional, the fisheye MEI camera data, defaults to None
        :param lidars: Optional, the LiDAR data, defaults to None
        :param scenario_tags: Optional, the scenario tags, defaults to None
        :param route_lane_group_ids: Optional, the route lane group IDs, defaults to None
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the log writer and finalizes the log io operations."""


@dataclass
class LiDARData:
    """Helper dataclass to pass LiDAR data to log writers."""

    lidar_type: LiDARType

    timestamp: Optional[TimePoint] = None
    iteration: Optional[int] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None
    point_cloud: Optional[npt.NDArray[np.float32]] = None

    def __post_init__(self):
        assert self.has_file_path or self.has_point_cloud, (
            "Either file path (dataset_root and relative_path) or point_cloud must be provided for LiDARData."
        )

    @property
    def has_file_path(self) -> bool:
        return self.dataset_root is not None and self.relative_path is not None

    @property
    def has_point_cloud(self) -> bool:
        return self.point_cloud is not None


@dataclass
class CameraData:
    """Helper dataclass to pass Camera data to log writers."""

    camera_type: Union[PinholeCameraType, FisheyeMEICameraType]
    extrinsic: PoseSE3

    timestamp: Optional[TimePoint] = None
    jpeg_binary: Optional[bytes] = None
    numpy_image: Optional[npt.NDArray[np.uint8]] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        assert self.has_file_path or self.has_jpeg_binary or self.has_numpy_image, (
            "Either file path (dataset_root and relative_path) or jpeg_binary or numpy_image must be provided for CameraData."
        )

        if self.has_file_path:
            absolute_path = Path(self.dataset_root) / self.relative_path
            assert absolute_path.exists(), f"Camera file not found: {absolute_path}"

    @property
    def has_file_path(self) -> bool:
        return self.dataset_root is not None and self.relative_path is not None

    @property
    def has_jpeg_file_path(self) -> bool:
        return self.relative_path is not None and str(self.relative_path).lower().endswith((".jpg", ".jpeg"))

    @property
    def has_png_file_path(self) -> bool:
        return self.relative_path is not None and str(self.relative_path).lower().endswith((".png",))

    @property
    def has_jpeg_binary(self) -> bool:
        return self.jpeg_binary is not None

    @property
    def has_numpy_image(self) -> bool:
        return self.numpy_image is not None
