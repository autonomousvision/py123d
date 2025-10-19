from __future__ import annotations

from typing import Literal
from dataclasses import dataclass


@dataclass
class DatasetConverterConfig:

    force_log_conversion: bool = False
    force_map_conversion: bool = False

    # Map
    include_map: bool = False

    # Ego
    include_ego: bool = False

    # Box Detections
    include_box_detections: bool = False

    # Traffic Lights
    include_traffic_lights: bool = False

    # Cameras
    include_cameras: bool = False
    camera_store_option: Literal["path", "binary", "mp4"] = "path"

    # LiDARs
    include_lidars: bool = False
    lidar_store_option: Literal["path", "binary"] = "path"

    # Scenario tag / Route
    # NOTE: These are only supported for nuPlan. Consider removing or expanding support.
    include_scenario_tags: bool = False
    include_route: bool = False

    def __post_init__(self):
        assert self.camera_store_option != "mp4", "MP4 format is not yet supported, but planned for future releases."
        assert self.camera_store_option in [
            "path",
            "binary",
        ], f"Invalid camera store option, got {self.camera_store_option}."

        assert self.lidar_store_option in [
            "path",
            "binary",
        ], f"Invalid LiDAR store option, got {self.lidar_store_option}."
