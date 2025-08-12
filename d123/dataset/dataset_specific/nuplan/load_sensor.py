import io
from pathlib import Path

from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

from d123.common.datatypes.sensor.lidar import LiDAR, LiDARMetadata


def load_nuplan_lidar_from_path(filepath: Path, lidar_metadata: LiDARMetadata) -> LiDAR:
    assert filepath.exists(), f"LiDAR file not found: {filepath}"
    with open(filepath, "rb") as fp:
        buffer = io.BytesIO(fp.read())
    return LiDAR(metadata=lidar_metadata, point_cloud=LidarPointCloud.from_buffer(buffer, "pcd").points)


# def load_camera_from_path(filename: str, metadata: CameraMetadata) -> Camera:
#     camera_full_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "sensor_blobs" / filename
#     assert camera_full_path.exists(), f"Camera file not found: {camera_full_path}"
#     img = Image.open(camera_full_path)
#     img.load()
#     return Camera(metadata=metadata, image=np.asarray(img, dtype=np.uint8))
