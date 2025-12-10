from typing import Any

import msgpack
import numpy as np


def encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"dtype": obj.dtype.str, b"shape": obj.shape, b"data": obj.tobytes()}
    return obj


def decode_numpy(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.frombuffer(obj[b"data"], dtype=obj[b"dtype"]).reshape(obj[b"shape"])
    return obj


def msgpack_encode_with_numpy(data: Any) -> bytes:
    """Serialize a dictionary containing numpy arrays using msgpack.

    :param data: The dictionary to serialize.
    :return: The serialized bytes.
    """
    return msgpack.packb(data, default=encode_numpy, use_bin_type=True)  # type: ignore


def msgpack_decode_with_numpy(packed_data: bytes) -> dict:
    """Deserialize a dictionary containing numpy arrays using msgpack.

    :param packed_data: The serialized bytes.
    :return: The deserialized dictionary.
    """
    return msgpack.unpackb(packed_data, object_hook=decode_numpy, raw=False)  # type: ignore
