from pathlib import Path

import numpy as np

from d123.common.datatypes.sensor.lidar import LiDAR


def load_carla_lidar_from_path(filepath: Path) -> LiDAR:
    assert filepath.exists(), f"LiDAR file not found: {filepath}"
    return LiDAR(point_cloud=np.load(filepath))
