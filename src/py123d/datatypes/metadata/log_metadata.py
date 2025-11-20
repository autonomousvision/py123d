from __future__ import annotations

from typing import Dict, Optional, Type

import py123d
from py123d.conversion.registry.box_detection_label_registry import BOX_DETECTION_LABEL_REGISTRY, BoxDetectionLabel
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDARMetadata, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeCameraType
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


class LogMetadata:
    """Class to hold metadata information about a log."""

    __slots__ = (
        "_dataset",
        "_split",
        "_log_name",
        "_location",
        "_timestep_seconds",
        "_vehicle_parameters",
        "_box_detection_label_class",
        "_pinhole_camera_metadata",
        "_fisheye_mei_camera_metadata",
        "_lidar_metadata",
        "_map_metadata",
        "_version",
    )

    def __init__(
        self,
        dataset: str,
        split: str,
        log_name: str,
        location: str,
        timestep_seconds: float,
        vehicle_parameters: Optional[VehicleParameters] = None,
        box_detection_label_class: Optional[Type[BoxDetectionLabel]] = None,
        pinhole_camera_metadata: Optional[Dict[PinholeCameraType, PinholeCameraMetadata]] = {},
        fisheye_mei_camera_metadata: Optional[Dict[FisheyeMEICameraType, FisheyeMEICameraMetadata]] = {},
        lidar_metadata: Optional[Dict[LiDARType, LiDARMetadata]] = {},
        map_metadata: Optional[MapMetadata] = None,
        version: str = str(py123d.__version__),
    ):
        """Create a :class:`LogMetadata` instance from a dictionary.

        :param dataset: The dataset name in lowercase.
        :param split: Data split name, typically ``{dataset_name}_{train/val/test}``.
        :param log_name: Name of the log file.
        :param location: Location of the log data.
        :param timestep_seconds: The time interval between consecutive frames in seconds.
        :param vehicle_parameters: The :class:`~py123d.datatypes.vehicle_state.VehicleParameters`
            of the ego vehicle, if available.
        :param box_detection_label_class: The box detection label class specific to the dataset, if available.
        :param pinhole_camera_metadata: Dictionary of :class:`~py123d.datatypes.sensors.PinholeCameraType`
            to :class:`~py123d.datatypes.sensors.PinholeCameraMetadata`, defaults to {}
        :param fisheye_mei_camera_metadata: Dictionary of :class:`~py123d.datatypes.sensors.FisheyeMEICameraType`
            to :class:`~py123d.datatypes.sensors.FisheyeMEICameraMetadata`, defaults to {}
        :param lidar_metadata: Dictionary of :class:`~py123d.datatypes.sensors.LiDARType`
            to :class:`~py123d.datatypes.sensors.LiDARMetadata`, defaults to {}
        :param map_metadata: The :class:`~py123d.datatypes.metadata.MapMetadata` for the log, if available, defaults to None
        :param version: The version of the log metadata, defaults to str(py123d.__version__)
        """
        self._dataset = dataset
        self._split = split
        self._log_name = log_name
        self._location = location
        self._timestep_seconds = timestep_seconds
        self._vehicle_parameters = vehicle_parameters
        self._box_detection_label_class = box_detection_label_class
        self._pinhole_camera_metadata = pinhole_camera_metadata
        self._fisheye_mei_camera_metadata = fisheye_mei_camera_metadata
        self._lidar_metadata = lidar_metadata
        self._map_metadata = map_metadata
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
    def location(self) -> str:
        """Location of the log data."""
        return self._location

    @property
    def timestep_seconds(self) -> float:
        """The time interval between consecutive frames in seconds."""
        return self._timestep_seconds

    @property
    def vehicle_parameters(self) -> Optional[VehicleParameters]:
        """The :class:`~py123d.datatypes.vehicle_state.VehicleParameters` of the ego vehicle, if available."""
        return self._vehicle_parameters

    @property
    def box_detection_label_class(self) -> Optional[Type[BoxDetectionLabel]]:
        """The box detection label class specific to the dataset, if available."""
        return self._box_detection_label_class

    @property
    def pinhole_camera_metadata(self) -> Dict[PinholeCameraType, PinholeCameraMetadata]:
        """Dictionary of :class:`~py123d.datatypes.sensors.PinholeCameraType`
        to :class:`~py123d.datatypes.sensors.PinholeCameraMetadata`.
        """
        return self._pinhole_camera_metadata

    @property
    def fisheye_mei_camera_metadata(self) -> Dict[FisheyeMEICameraType, FisheyeMEICameraMetadata]:
        """Dictionary of :class:`~py123d.datatypes.sensors.FisheyeMEICameraType`
        to :class:`~py123d.datatypes.sensors.FisheyeMEICameraMetadata`.
        """
        return self._fisheye_mei_camera_metadata

    @property
    def lidar_metadata(self) -> Dict[LiDARType, LiDARMetadata]:
        """Dictionary of :class:`~py123d.datatypes.sensors.LiDARType`
        to :class:`~py123d.datatypes.sensors.LiDARMetadata`.
        """
        return self._lidar_metadata

    @property
    def map_metadata(self) -> Optional[MapMetadata]:
        """The :class:`~py123d.datatypes.metadata.MapMetadata` associated with the log, if available."""
        return self._map_metadata

    @property
    def version(self) -> str:
        """Version of the py123d library used to create this log metadata (not used currently)."""
        return self._version

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:
        """Create a :class:`LogMetadata` instance from a Python dictionary.

        :param data_dict: Dictionary containing log metadata.
        :raises ValueError: If the dictionary is missing required fields.
        :return: A :class:`LogMetadata` instance.
        """

        # Ego Vehicle Parameters
        if data_dict["vehicle_parameters"] is not None:
            data_dict["vehicle_parameters"] = VehicleParameters.from_dict(data_dict["vehicle_parameters"])

        # Box detection label class specific to the dataset
        if data_dict["box_detection_label_class"] in BOX_DETECTION_LABEL_REGISTRY:
            data_dict["box_detection_label_class"] = BOX_DETECTION_LABEL_REGISTRY[
                data_dict["box_detection_label_class"]
            ]
        elif data_dict["box_detection_label_class"] is None:
            data_dict["box_detection_label_class"] = None
        else:
            raise ValueError(f"Unknown box detection label class: {data_dict['box_detection_label_class']}")

        # Pinhole Camera Metadata
        pinhole_camera_metadata = {}
        for key, value in data_dict.get("pinhole_camera_metadata", {}).items():
            pinhole_camera_metadata[PinholeCameraType.deserialize(key)] = PinholeCameraMetadata.from_dict(value)
        data_dict["pinhole_camera_metadata"] = pinhole_camera_metadata

        # Fisheye MEI Camera Metadata
        fisheye_mei_camera_metadata = {}
        for key, value in data_dict.get("fisheye_mei_camera_metadata", {}).items():
            fisheye_mei_camera_metadata[FisheyeMEICameraType.deserialize(key)] = FisheyeMEICameraMetadata.from_dict(
                value
            )
        data_dict["fisheye_mei_camera_metadata"] = fisheye_mei_camera_metadata

        # LiDAR Metadata
        data_dict["lidar_metadata"] = {
            LiDARType.deserialize(key): LiDARMetadata.from_dict(value)
            for key, value in data_dict.get("lidar_metadata", {}).items()
        }

        # Map Metadata
        if data_dict["map_metadata"] is not None:
            data_dict["map_metadata"] = MapMetadata.from_dict(data_dict["map_metadata"])

        return LogMetadata(**data_dict)

    def to_dict(self) -> Dict:
        """Convert the :class:`LogMetadata` instance to a Python dictionary.

        :return: A dictionary representation of the log metadata.
        """
        data_dict = {slot.lstrip("_"): getattr(self, slot) for slot in self.__slots__}

        # Override complex types with their dictionary representations
        data_dict["vehicle_parameters"] = self.vehicle_parameters.to_dict() if self.vehicle_parameters else None
        if self.box_detection_label_class is not None:
            data_dict["box_detection_label_class"] = self.box_detection_label_class.__name__
        data_dict["pinhole_camera_metadata"] = {
            key.serialize(): value.to_dict() for key, value in self.pinhole_camera_metadata.items()
        }
        data_dict["fisheye_mei_camera_metadata"] = {
            key.serialize(): value.to_dict() for key, value in self.fisheye_mei_camera_metadata.items()
        }
        data_dict["lidar_metadata"] = {key.serialize(): value.to_dict() for key, value in self.lidar_metadata.items()}
        data_dict["map_metadata"] = self.map_metadata.to_dict() if self.map_metadata else None
        return data_dict

    def __repr__(self) -> str:
        return (
            f"LogMetadata(dataset={self.dataset}, split={self.split}, log_name={self.log_name}, "
            f"location={self.location}, timestep_seconds={self.timestep_seconds}, "
            f"vehicle_parameters={self.vehicle_parameters}, "
            f"box_detection_label_class={self.box_detection_label_class}, "
            f"pinhole_camera_metadata={self.pinhole_camera_metadata}, "
            f"fisheye_mei_camera_metadata={self.fisheye_mei_camera_metadata}, "
            f"lidar_metadata={self.lidar_metadata}, map_metadata={self.map_metadata}, "
            f"version={self.version})"
        )
