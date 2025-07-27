import io
import os
from pathlib import Path

from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

from d123.common.datatypes.sensor.lidar import LiDAR

NUPLAN_DATA_ROOT = Path(os.environ["NUPLAN_DATA_ROOT"])


def load_lidar_from_path(filename: str) -> LiDAR:
    lidar_full_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "sensor_blobs" / filename

    assert lidar_full_path.exists(), f"LiDAR file not found: {lidar_full_path}"

    with open(lidar_full_path, "rb") as fp:
        buffer = io.BytesIO(fp.read())

    points = LidarPointCloud.from_buffer(buffer, "pcd").points
    return LiDAR(point_cloud=points)  # Reshape to
