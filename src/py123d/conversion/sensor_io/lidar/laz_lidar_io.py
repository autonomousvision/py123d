import io

import laspy
import numpy as np
import numpy.typing as npt


def is_laz_binary(laz_binary: bytes) -> bool:
    """Check if the given binary data represents a LAZ compressed point cloud.

    :param laz_binary: The binary data to check.
    :return: True if the binary data is a LAZ compressed point cloud, False otherwise.
    """
    LAS_MAGIC_NUMBER = b"LASF"
    return laz_binary[0:4] == LAS_MAGIC_NUMBER


def encode_point_cloud_3d_as_laz_binary(point_cloud: npt.NDArray[np.float32]) -> bytes:
    """Compress Lidar point cloud data using LAZ format.

    :param point_cloud: The Lidar point cloud data to compress, as numpy array.
    :return: The compressed LAZ binary data.
    """

    # Create a LAS file in memory, and populate it with point cloud data
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = point_cloud[:, 0]
    las.y = point_cloud[:, 1]
    las.z = point_cloud[:, 2]

    # Write to memory buffer and return compressed binary data
    buffer = io.BytesIO()
    las.write(buffer, do_compress=True)
    laz_binary = buffer.getvalue()

    return laz_binary


def load_point_cloud_3d_from_laz_binary(laz_binary: bytes) -> npt.NDArray[np.float32]:
    """Decompress Lidar point cloud data from LAZ format.

    :param laz_binary: The compressed LAZ binary data.
    :param lidar_metadata: Metadata associated with the Lidar data.
    :raises ValueError: If the Lidar features are not found in the LAS file.
    :return: The decompressed Lidar point cloud data as a numpy array.
    """

    # Read the LAS file from memory buffer
    buffer = io.BytesIO(laz_binary)
    las = laspy.read(buffer)

    # Extract the point cloud data
    lidar_pc = np.array(las.xyz, dtype=np.float32)
    assert lidar_pc.ndim == 2, "Lidar point cloud must be a 2D array of shape (N, 3) for LAZ decompression."
    assert lidar_pc.shape[-1] == 3, "Lidar point cloud must have 3 attributes (x, y, z) for LAZ decompression."
    return lidar_pc
