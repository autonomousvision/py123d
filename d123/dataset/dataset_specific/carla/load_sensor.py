from pathlib import Path

import numpy as np

from d123.common.datatypes.sensor.lidar import LiDAR, LiDARMetadata


def load_carla_lidar_from_path(filepath: Path, lidar_metadata: LiDARMetadata) -> LiDAR:
    assert filepath.exists(), f"LiDAR file not found: {filepath}"
    return LiDAR(metadata=lidar_metadata, point_cloud=np.load(filepath))
