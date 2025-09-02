import io
from pathlib import Path


from d123.common.utils.dependencies import check_dependencies
from d123.common.datatypes.sensor.lidar import LiDAR, LiDARMetadata

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud


def load_nuplan_lidar_from_path(filepath: Path, lidar_metadata: LiDARMetadata) -> LiDAR:
    assert filepath.exists(), f"LiDAR file not found: {filepath}"
    with open(filepath, "rb") as fp:
        buffer = io.BytesIO(fp.read())
    return LiDAR(metadata=lidar_metadata, point_cloud=LidarPointCloud.from_buffer(buffer, "pcd").points)

