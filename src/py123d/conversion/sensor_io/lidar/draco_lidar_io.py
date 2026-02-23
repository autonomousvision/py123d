from typing import Final

import DracoPy
import numpy as np
import numpy.typing as npt

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


def encode_point_cloud_3d_as_draco_binary(lidar_pc: npt.NDArray[np.float32]) -> bytes:
    """Compress Lidar point cloud data using Draco format.

    :param point_cloud: The Lidar point cloud data to compress, as numpy array.
    :return: The compressed Draco binary data.
    """
    assert lidar_pc.ndim == 2, "Lidar point cloud must be a 2D array of shape (N, 3) for Draco compression."
    assert lidar_pc.shape[-1] == 3, "Lidar point cloud must have 3 attributes (x, y, z) for Draco compression."
    # TODO: Add variable dtypes, other than float32.
    return DracoPy.encode(
        lidar_pc,
        quantization_bits=DRACO_QUANTIZATION_BITS,
        compression_level=DRACO_COMPRESSION_LEVEL,
        quantization_range=DRACO_QUANTIZATION_RANGE,
        quantization_origin=None,
        create_metadata=False,
        preserve_order=DRACO_PRESERVE_ORDER,
    )


def load_point_cloud_3d_from_draco_binary(draco_binary: bytes) -> npt.NDArray[np.float32]:
    """Decompress Lidar point cloud data from Draco format.

    :param draco_binary: The compressed Draco binary data.
    :raises ValueError: If the Lidar features are not found in the decompressed data.
    :return: The decompressed Lidar point cloud data as a numpy array.
    """
    # TODO: Add variable dtypes, other than float32.
    mesh = DracoPy.decode(draco_binary)
    lidar_pc = np.array(mesh.points, dtype=np.float32)
    assert lidar_pc.ndim == 2, "Lidar point cloud must be a 2D array of shape (N, 3) for Draco compression."
    assert lidar_pc.shape[-1] == 3, "Lidar point cloud must have 3 attributes (x, y, z) for Draco compression."
    return lidar_pc
