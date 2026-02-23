from typing import Literal, Optional

import numpy as np
import pyarrow as pa

from py123d.geometry.geometry_index import Point3DIndex


def is_ipc_binary(blob: bytes) -> bool:
    """Check if the given binary data represents an Arrow IPC stream.

    :param blob: The binary data to check.
    :return: True if the binary data is an Arrow IPC stream, False otherwise.
    """
    ARROW_MAGIC_NUMBER = b"ARROW1"
    return blob.startswith(ARROW_MAGIC_NUMBER)


def encode_point_cloud_3d_as_ipc_binary(
    point_cloud: np.ndarray, codec: Optional[Literal["zstd", "lz4"]] = "zstd"
) -> bytes:
    """Compresses a Lidar point cloud (as a numpy array) into an Arrow IPC binary blob, using the specified codec.

    :param point_cloud: The Lidar point cloud data to compress, as a numpy array of shape (N, 3).
    :param codec: The compression codec to use, either "zstd" or "lz4", defaults to "zstd".
    :return: The compressed Arrow IPC binary data.
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[1] == len(Point3DIndex), (
        "Lidar point cloud must be a 2-dim array of shape (N, 3)."
    )
    # NOTE @DanielDauner: Here we just used the features function, for simplicity.
    pc_dict = {
        "x": point_cloud[:, Point3DIndex.X],
        "y": point_cloud[:, Point3DIndex.Y],
        "z": point_cloud[:, Point3DIndex.Z],
    }
    return encode_point_cloud_features_as_ipc_binary(pc_dict, codec=codec)


def load_point_cloud_3d_from_ipc_binary(blob: bytes) -> np.ndarray:
    """Decompresses an Arrow IPC binary blob back into a Lidar point cloud numpy array of shape (N, 3).

    :param blob: The compressed Arrow IPC binary data containing the Lidar point cloud features.
    :return: The decompressed Lidar point cloud data as a numpy array of shape (N, 3).
    """
    feature_dict = load_point_cloud_features_from_ipc_binary(blob)
    point_cloud = np.stack((feature_dict["x"], feature_dict["y"], feature_dict["z"]), axis=-1)
    assert point_cloud.ndim == 2 and point_cloud.shape[1] == len(Point3DIndex), (
        f"Decoded Lidar point cloud must be a 2-dim array of shape (N, 3). Got shape {point_cloud.shape}"
    )
    return point_cloud


def encode_point_cloud_features_as_ipc_binary(
    feature_dict: dict[str, np.ndarray], codec: Optional[Literal["zstd", "lz4"]] = "zstd"
) -> bytes:
    """Compresses a dictionary of Lidar point cloud features (as numpy arrays) into an Arrow IPC binary blob, \
        using the specified codec.

    :param feature_dict: The dictionary of Lidar point cloud features to compress.
    :param codec: The compression codec to use, either "zstd" or "lz4", defaults to "zstd".
    :return: The compressed Arrow IPC binary data.
    """
    batch = pa.RecordBatch.from_pydict(feature_dict)
    sink = pa.BufferOutputStream()

    # NOTE @DanielDauner: The IPC writer options could be further tuned.
    options = pa.ipc.IpcWriteOptions(compression=codec)
    with pa.ipc.new_stream(sink, batch.schema, options=options) as writer:
        writer.write_batch(batch)

    return sink.getvalue().to_pybytes()


def load_point_cloud_features_from_ipc_binary(blob: bytes) -> dict[str, np.ndarray]:
    """Decompresses an Arrow IPC binary blob back into a dictionary of Lidar point cloud features as numpy arrays.

    :param blob: The compressed Arrow IPC binary data containing the Lidar point cloud features.
    :return: The decompressed Lidar point cloud features as a dictionary of numpy arrays.
    """
    buffer = pa.BufferReader(blob)
    with pa.ipc.open_stream(buffer) as reader:
        batch = reader.read_next_batch()
    return {col: batch.column(i).to_numpy(zero_copy_only=False) for i, col in enumerate(batch.schema.names)}
