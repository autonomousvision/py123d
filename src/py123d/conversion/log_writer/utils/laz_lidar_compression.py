import io

import laspy
import numpy as np
import numpy.typing as npt

from py123d.datatypes.sensors.lidar.lidar import LiDAR, LiDARMetadata


def compress_lidar_with_laz(point_cloud: npt.NDArray[np.float32], lidar_metadata: LiDARMetadata) -> bytes:
    """Compress LiDAR point cloud data using LAZ format.

    :param point_cloud: The LiDAR point cloud data to compress, as numpy array.
    :param lidar_metadata: Metadata associated with the LiDAR data.
    :return: The compressed LAZ binary data.
    """

    lidar_index = lidar_metadata.lidar_index

    # Create a LAS file in memory, and populate it with point cloud data
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = point_cloud[:, lidar_index.X]
    las.y = point_cloud[:, lidar_index.Y]
    las.z = point_cloud[:, lidar_index.Z]

    # Add additional LiDAR features if present
    for feature in lidar_metadata.lidar_index:
        if feature.name in ["X", "Y", "Z"]:
            continue  # Already saved above
        las.add_extra_dim(laspy.ExtraBytesParams(name=feature.name, type="float32"))
        las[feature.name] = point_cloud[:, feature.value]

    # Write to memory buffer and return compressed binary data
    buffer = io.BytesIO()
    las.write(buffer, do_compress=True)
    laz_binary = buffer.getvalue()

    return laz_binary


def decompress_lidar_from_laz(laz_binary: bytes, lidar_metadata: LiDARMetadata) -> npt.NDArray[np.float32]:
    """Decompress LiDAR point cloud data from LAZ format.

    :param laz_binary: The compressed LAZ binary data.
    :param lidar_metadata: Metadata associated with the LiDAR data.
    :raises ValueError: If the LiDAR features are not found in the LAS file.
    :return: The decompressed LiDAR point cloud data as a numpy array.
    """

    lidar_index = lidar_metadata.lidar_index

    # Read the LAS file from memory buffer
    buffer = io.BytesIO(laz_binary)
    las = laspy.read(buffer)

    # Extract the point cloud data
    xyz = las.xyz

    num_points = len(xyz)
    point_cloud = np.zeros((num_points, len(lidar_metadata.lidar_index)), dtype=np.float32)
    point_cloud[:, lidar_index.XYZ] = xyz

    for feature in lidar_index:
        if feature.name in ["X", "Y", "Z"]:
            continue  # Already loaded above
        if hasattr(las, feature.name):
            point_cloud[:, feature.value] = las[feature.name]
        else:
            raise ValueError(f"LiDAR feature {feature.name} not found in LAS file.")

    return LiDAR(point_cloud=point_cloud, metadata=lidar_metadata)
