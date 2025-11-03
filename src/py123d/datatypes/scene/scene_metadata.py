from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import py123d
from py123d.datatypes.maps.map_metadata import MapMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDARMetadata, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeCameraType
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


@dataclass
class LogMetadata:

    dataset: str
    split: str
    log_name: str
    location: str
    timestep_seconds: float

    vehicle_parameters: Optional[VehicleParameters] = None
    pinhole_camera_metadata: Dict[PinholeCameraType, PinholeCameraMetadata] = field(default_factory=dict)
    fisheye_mei_camera_metadata: Dict[FisheyeMEICameraType, FisheyeMEICameraMetadata] = field(default_factory=dict)
    lidar_metadata: Dict[LiDARType, LiDARMetadata] = field(default_factory=dict)

    map_metadata: Optional[MapMetadata] = None
    version: str = str(py123d.__version__)

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:

        # Ego Vehicle Parameters
        if data_dict["vehicle_parameters"] is not None:
            data_dict["vehicle_parameters"] = VehicleParameters.from_dict(data_dict["vehicle_parameters"])

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
        data_dict = asdict(self)
        data_dict["vehicle_parameters"] = self.vehicle_parameters.to_dict() if self.vehicle_parameters else None
        data_dict["pinhole_camera_metadata"] = {
            key.serialize(): value.to_dict() for key, value in self.pinhole_camera_metadata.items()
        }
        data_dict["fisheye_mei_camera_metadata"] = {
            key.serialize(): value.to_dict() for key, value in self.fisheye_mei_camera_metadata.items()
        }
        data_dict["lidar_metadata"] = {key.serialize(): value.to_dict() for key, value in self.lidar_metadata.items()}
        data_dict["map_metadata"] = self.map_metadata.to_dict() if self.map_metadata else None
        return data_dict


@dataclass(frozen=True)
class SceneExtractionMetadata:

    initial_uuid: str
    initial_idx: int
    duration_s: float
    history_s: float
    iteration_duration_s: float

    @property
    def number_of_iterations(self) -> int:
        return round(self.duration_s / self.iteration_duration_s)

    @property
    def number_of_history_iterations(self) -> int:
        return round(self.history_s / self.iteration_duration_s)

    @property
    def end_idx(self) -> int:
        return self.initial_idx + self.number_of_iterations
