from __future__ import annotations

from typing import Dict, List, Optional

import py123d
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
from py123d.datatypes.metadata.base_metadata import BaseMetadata, BaseModalityMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID, PinholeCameraMetadata
from py123d.datatypes.vehicle_state.ego_metadata import EgoStateSE3Metadata


class LogMetadata(BaseMetadata):
    """Class to hold metadata information about a log."""

    __slots__ = (
        "_dataset",
        "_split",
        "_log_name",
        "_location",
        "_timestep_seconds",
        "_map_metadata",
        "_ego_state_se3_metadata",
        "_box_detections_se3_metadata",
        "_traffic_light_detections_metadata",
        "_pinhole_cameras_metadata",
        "_lidars_metadata",
        "_lidar_merged_metadata",
        "_custom_modalities_metadata",
        "_version",
    )

    def __init__(
        self,
        dataset: str,
        split: str,
        log_name: str,
        location: Optional[str],
        timestep_seconds: float,
        map_metadata: Optional[MapMetadata] = None,
        ego_state_se3_metadata: Optional[EgoStateSE3Metadata] = None,
        box_detections_se3_metadata: Optional[BoxDetectionsSE3Metadata] = None,
        traffic_light_detections_metadata: Optional[TrafficLightDetectionsMetadata] = None,
        pinhole_cameras_metadata: Optional[Dict[PinholeCameraID, PinholeCameraMetadata]] = None,
        lidars_metadata: Optional[Dict[LidarID, LidarMetadata]] = None,
        lidar_merged_metadata: Optional[LidarMergedMetadata] = None,
        custom_modalities_metadata: Optional[Dict[str, CustomModalityMetadata]] = None,
        version: str = str(py123d.__version__),
    ):
        """Create a :class:`LogMetadata` instance from a dictionary.

        :param dataset: The dataset name in lowercase.
        :param split: Data split name, typically ``{dataset_name}_{train/val/test}``.
        :param log_name: Name of the log file.
        :param location: Location of the log data.
        :param timestep_seconds: The time interval between consecutive frames in seconds.
        """

        # Basic log info
        self._dataset = dataset
        self._split = split
        self._log_name = log_name
        self._location = location
        self._timestep_seconds = timestep_seconds

        # Map metadata
        self._map_metadata: Optional[MapMetadata] = map_metadata

        # Modality Meta
        self._ego_state_se3_metadata: Optional[EgoStateSE3Metadata] = ego_state_se3_metadata
        self._box_detections_se3_metadata: Optional[BoxDetectionsSE3Metadata] = box_detections_se3_metadata
        self._traffic_light_detections_metadata: Optional[TrafficLightDetectionsMetadata] = (
            traffic_light_detections_metadata
        )
        self._pinhole_cameras_metadata: Optional[Dict[PinholeCameraID, PinholeCameraMetadata]] = (
            pinhole_cameras_metadata
        )
        self._lidars_metadata: Optional[Dict[LidarID, LidarMetadata]] = lidars_metadata
        self._lidar_merged_metadata: Optional[LidarMergedMetadata] = lidar_merged_metadata
        self._custom_modalities_metadata: Optional[Dict[str, CustomModalityMetadata]] = custom_modalities_metadata

        # Currently not used, but can be helpful for tracking library version used to create the log metadata
        self._version = version

    @property
    def dataset(self) -> str:
        """The dataset name in lowercase."""
        return self._dataset

    @property
    def split(self) -> str:
        """Data split name, typically ``{dataset_name}_{train/val/test}``."""
        return self._split

    @property
    def log_name(self) -> str:
        """Name of the log file."""
        return self._log_name

    @property
    def location(self) -> Optional[str]:
        """Location of the log data."""
        return self._location

    @property
    def timestep_seconds(self) -> float:
        """The time interval between consecutive frames in seconds."""
        return self._timestep_seconds

    @property
    def version(self) -> str:
        """Version of the py123d library used to create this log metadata (not used currently)."""
        return self._version

    @property
    def all_modality_metadatas(self) -> List[BaseModalityMetadata]:
        """Returns a flat list of all modality metadata present in this log."""
        result: List[BaseModalityMetadata] = []
        if self._ego_state_se3_metadata is not None:
            result.append(self._ego_state_se3_metadata)
        if self._box_detections_se3_metadata is not None:
            result.append(self._box_detections_se3_metadata)
        if self._traffic_light_detections_metadata is not None:
            result.append(self._traffic_light_detections_metadata)
        if self._pinhole_cameras_metadata is not None:
            for cam_meta in self._pinhole_cameras_metadata.values():
                result.append(cam_meta)
        if self._lidars_metadata is not None:
            for lidar_meta in self._lidars_metadata.values():
                result.append(lidar_meta)
        if self._lidar_merged_metadata is not None:
            result.append(self._lidar_merged_metadata)
        if self._custom_modalities_metadata is not None:
            for custom_meta in self._custom_modalities_metadata.values():
                result.append(custom_meta)
        return result

    # Fields that are JSON-serializable and stored in Arrow schema metadata.
    _SERIALIZED_FIELDS = ("dataset", "split", "log_name", "location", "timestep_seconds", "version")

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:
        """Create a :class:`LogMetadata` instance from a Python dictionary.

        Only the basic log fields are expected (the ones produced by :meth:`to_dict`).
        Modality metadata is stored separately in each modality's Arrow file.

        :param data_dict: Dictionary containing log metadata.
        :return: A :class:`LogMetadata` instance.
        """
        return LogMetadata(
            dataset=data_dict["dataset"],
            split=data_dict["split"],
            log_name=data_dict["log_name"],
            location=data_dict.get("location"),
            timestep_seconds=data_dict["timestep_seconds"],
            version=data_dict.get("version", "unknown"),
        )

    def to_dict(self) -> Dict:
        """Convert the :class:`LogMetadata` instance to a JSON-serializable dictionary.

        Only serializes basic log fields. Modality metadata is stored separately
        in each modality's own Arrow file schema.

        :return: A dictionary representation of the log metadata.
        """
        return {f: getattr(self, f"_{f}") for f in self._SERIALIZED_FIELDS}

    def __repr__(self) -> str:
        return (
            f"LogMetadata(dataset={self.dataset}, split={self.split}, log_name={self.log_name}, "
            f"location={self.location}, timestep_seconds={self.timestep_seconds}, "
            f"version={self.version})"
        )
