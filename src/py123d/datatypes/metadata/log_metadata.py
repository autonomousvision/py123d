from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Type

import py123d
from py123d.conversion.registry.box_detection_label_registry import BOX_DETECTION_LABEL_REGISTRY, BoxDetectionLabel
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDARMetadata, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType, PinholeMetadata
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


@dataclass
class LogMetadata:

    dataset: str
    split: str
    log_name: str
    location: str
    timestep_seconds: float

    vehicle_parameters: Optional[VehicleParameters] = None
    box_detection_label_class: Optional[Type[BoxDetectionLabel]] = None
    pinhole_camera_metadata: Dict[PinholeCameraType, PinholeMetadata] = field(default_factory=dict)
    fisheye_mei_camera_metadata: Dict[FisheyeMEICameraType, FisheyeMEICameraMetadata] = field(default_factory=dict)
    lidar_metadata: Dict[LiDARType, LiDARMetadata] = field(default_factory=dict)

    map_metadata: Optional[MapMetadata] = None
    version: str = str(py123d.__version__)

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:

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
            pinhole_camera_metadata[PinholeCameraType.deserialize(key)] = PinholeMetadata.from_dict(value)
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
        data_dict = asdict(self)
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
