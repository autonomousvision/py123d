from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import numpy.typing as npt

from py123d.datatypes import (
    BaseMapObject,
    BoxDetectionsSE3,
    CustomModality,
    EgoStateSE3,
    FisheyeMEICameraID,
    LidarID,
    LogMetadata,
    MapMetadata,
    PinholeCameraID,
    Timestamp,
    TrafficLightDetections,
)
from py123d.datatypes.metadata.base_metadata import BaseModalityMetadata
from py123d.geometry import PoseSE3


class DatasetParser(abc.ABC):
    """Top-level parser that produces per-log and per-map containers.

    An orchestrator calls :meth:`get_log_parsers` / :meth:`get_map_parsers` once on
    the main process, then distributes the resulting lightweight containers to workers.
    """

    @abc.abstractmethod
    def get_map_parsers(self) -> List[MapParser]:
        """Returns one :class:`MapParser` per map region in the dataset."""

    @abc.abstractmethod
    def get_log_parsers(self) -> List[LogParser]:
        """Returns one :class:`LogParser` per log in the dataset."""


class MapParser(abc.ABC):
    """Lightweight, picklable handle to one map's data."""

    @abc.abstractmethod
    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata describing this map (location, coordinate system, etc.)."""

    @abc.abstractmethod
    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Yields map objects lazily, one at a time."""


class LogParser(abc.ABC):
    """Lightweight, picklable handle to one log's data.

    Implementations hold only the paths and parameters needed to read the raw data.
    The heavy I/O happens lazily inside :meth:`iter_frames`.
    """

    @abc.abstractmethod
    def get_log_metadata(self) -> LogMetadata:
        pass

    @abc.abstractmethod
    def iter_frames(self) -> Iterator[ParsedFrame]:
        """Yields ``(timestamp, modalities)`` tuples.

        Each tuple contains the frame timestamp and a dict mapping modality names
        to their data objects. The orchestrator forwards these to
        ``writer.write_sync(timestamp, **modalities)``.
        """

    @abc.abstractmethod
    def iter_modality_async(self, modality_metadata: BaseModalityMetadata) -> Iterator[ParsedModality]:
        """Yields ``(timestamp, data)`` tuples for a single modality at its native rate."""


@dataclass
class ParsedFrame:
    """One synchronized frame of data, as produced by a :class:`LogParser`.

    Fields mirror the ``AbstractLogWriter.write()`` signature so that an orchestrator
    can forward them directly::

        for frame in log_parser.iter_frames():
            writer.write(**frame.to_writer_kwargs())
    """

    timestamp: Timestamp
    uuid: Optional[uuid.UUID] = None
    ego_state_se3: Optional[EgoStateSE3] = None
    box_detections_se3: Optional[BoxDetectionsSE3] = None
    traffic_lights: Optional[TrafficLightDetections] = None
    pinhole_cameras: Optional[List[ParsedCamera]] = None
    fisheye_mei_cameras: Optional[List[ParsedCamera]] = None
    lidars: Optional[List[ParsedLidar]] = None
    custom_modalities: Optional[Dict[str, CustomModality]] = None


@dataclass
class ParsedLidar:
    """Helper dataclass to pass Lidar data to log writers."""

    lidar_name: str
    lidar_type: LidarID
    start_timestamp: Timestamp
    end_timestamp: Timestamp

    iteration: Optional[int] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None
    point_cloud_3d: Optional[npt.NDArray] = None
    point_cloud_features: Optional[Dict[str, npt.NDArray]] = None

    def __post_init__(self):
        assert self.has_file_path or self.has_point_cloud_3d, (
            "Either file path (dataset_root and relative_path) or point_cloud must be provided for LidarData."
        )

    @property
    def has_file_path(self) -> bool:
        return self.dataset_root is not None and self.relative_path is not None

    @property
    def has_point_cloud_3d(self) -> bool:
        return self.point_cloud_3d is not None

    @property
    def has_point_cloud_features(self) -> bool:
        return self.point_cloud_features is not None


@dataclass
class ParsedCamera:
    """Helper dataclass to pass Camera data to log writers."""

    camera_name: str
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID]
    extrinsic: PoseSE3
    timestamp: Timestamp

    jpeg_binary: Optional[bytes] = None
    numpy_image: Optional[npt.NDArray[np.uint8]] = None
    dataset_root: Optional[Union[str, Path]] = None
    relative_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        assert self.has_file_path or self.has_jpeg_binary or self.has_numpy_image, (
            "Either file path (dataset_root and relative_path) or jpeg_binary or numpy_image must be provided for CameraData."
        )

        if self.has_file_path:
            absolute_path = Path(self.dataset_root) / self.relative_path  # type: ignore
            assert absolute_path.exists(), f"Camera file not found: {absolute_path}"

    @property
    def has_file_path(self) -> bool:
        return self.dataset_root is not None and self.relative_path is not None

    @property
    def has_jpeg_file_path(self) -> bool:
        return self.has_file_path and str(self.relative_path).lower().endswith((".jpg", ".jpeg"))

    @property
    def has_png_file_path(self) -> bool:
        return self.has_file_path and str(self.relative_path).lower().endswith((".png",))

    @property
    def has_jpeg_binary(self) -> bool:
        return self.jpeg_binary is not None

    @property
    def has_numpy_image(self) -> bool:
        return self.numpy_image is not None


ParsedModality = Union[EgoStateSE3, BoxDetectionsSE3, TrafficLightDetections, ParsedCamera, ParsedLidar, CustomModality]
