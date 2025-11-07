from typing import Final

import DracoPy
import numpy as np
import numpy.typing as npt

from py123d.datatypes.sensors.lidar import LiDAR, LiDARMetadata

# TODO: add to config
DRACO_QUANTIZATION_BITS: Final[int] = 14
DRACO_COMPRESSION_LEVEL: Final[int] = 10  # Range: 0 (fastest) to 10 (slowest, best compression)
DRACO_QUANTIZATION_RANGE: Final[int] = -1  # Use default range
DRACO_PRESERVE_ORDER: Final[bool] = False


def is_draco_binary(draco_binary: bytes) -> bool:
    """Check if the given binary data represents a Draco compressed point cloud.

    :param draco_binary: The binary data to check.
    :return: True if the binary data is a Draco compressed point cloud, False otherwise.
    """
    DRACO_MAGIC_NUMBER = b"DRACO"
    return draco_binary.startswith(DRACO_MAGIC_NUMBER)


def encode_lidar_pc_as_draco_binary(lidar_pc: npt.NDArray[np.float32], lidar_metadata: LiDARMetadata) -> bytes:
    """Compress LiDAR point cloud data using Draco format.

    :param point_cloud: The LiDAR point cloud data to compress, as numpy array.
    :param lidar_metadata: Metadata associated with the LiDAR data.
    :return: The compressed Draco binary data.
    """
    lidar_index = lidar_metadata.lidar_index

    binary = DracoPy.encode(
        lidar_pc[:, lidar_index.XYZ],
        quantization_bits=DRACO_QUANTIZATION_BITS,
        compression_level=DRACO_COMPRESSION_LEVEL,
        quantization_range=DRACO_QUANTIZATION_RANGE,
        quantization_origin=None,
        create_metadata=False,
        preserve_order=DRACO_PRESERVE_ORDER,
    )

    return binary


def load_lidar_from_draco_binary(draco_binary: bytes, lidar_metadata: LiDARMetadata) -> LiDAR:
    """Decompress LiDAR point cloud data from Draco format.

    :param draco_binary: The compressed Draco binary data.
    :param lidar_metadata: Metadata associated with the LiDAR data.
    :raises ValueError: If the LiDAR features are not found in the decompressed data.
    :return: The decompressed LiDAR point cloud data as a LiDAR object.
    """

    # Decompress using Draco
    mesh = DracoPy.decode(draco_binary)
    points = mesh.points
    points = np.array(points, dtype=np.float32)

    return LiDAR(point_cloud=points, metadata=lidar_metadata)
