from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import d123
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraMetadata, PinholeCameraType
from d123.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
from d123.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


@dataclass
class LogMetadata:

    dataset: str
    split: str
    log_name: str
    location: str
    timestep_seconds: float

    vehicle_parameters: VehicleParameters
    camera_metadata: Dict[PinholeCameraType, PinholeCameraMetadata]
    lidar_metadata: Dict[LiDARType, LiDARMetadata]

    map_has_z: bool
    map_is_local: bool
    version: str = str(d123.__version__)

    @classmethod
    def from_dict(cls, data_dict: Dict) -> LogMetadata:

        data_dict["vehicle_parameters"] = VehicleParameters.from_dict(data_dict["vehicle_parameters"])
        data_dict["camera_metadata"] = {
            PinholeCameraType.deserialize(key): PinholeCameraMetadata.from_dict(value)
            for key, value in data_dict.get("camera_metadata", {}).items()
        }
        data_dict["lidar_metadata"] = {
            LiDARType.deserialize(key): LiDARMetadata.from_dict(value)
            for key, value in data_dict.get("lidar_metadata", {}).items()
        }
        return LogMetadata(**data_dict)

    def to_dict(self) -> Dict:
        data_dict = asdict(self)
        data_dict["vehicle_parameters"] = self.vehicle_parameters.to_dict()
        data_dict["camera_metadata"] = {key.serialize(): value.to_dict() for key, value in self.camera_metadata.items()}
        data_dict["lidar_metadata"] = {key.serialize(): value.to_dict() for key, value in self.lidar_metadata.items()}
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
