"""Lidar point cloud reader for the nuReasoning dataset.

nuReasoning point clouds are stored as ``.pcd`` files. Unlike nuPlan (which uses uncompressed
``binary`` PCDs), nuReasoning uses PCL ``binary_compressed`` (LZF-compressed, structure-of-arrays)
PCDs with the fields::

    x y z intensity lidar_info ring azimuth range is_second_return lidar_confidence

The merged point cloud combines multiple lidar sensors; the ``lidar_info`` field encodes which
sensor each point came from (mapped to :class:`LidarID` via ``NUREASONING_LIDAR_DICT``).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from py123d.datatypes.sensors import LidarFeature
from py123d.parser.nureasoning.utils.nureasoning_constants import NUREASONING_LIDAR_DICT

# PCD (size, type_char) -> little-endian numpy dtype string.
_PCD_TYPE_MAP: Dict[Tuple[int, str], str] = {
    (1, "I"): "i1",
    (1, "U"): "u1",
    (2, "I"): "<i2",
    (2, "U"): "<u2",
    (4, "I"): "<i4",
    (4, "U"): "<u4",
    (4, "F"): "<f4",
    (8, "F"): "<f8",
    (8, "I"): "<i8",
    (8, "U"): "<u8",
}


def _parse_pcd_header(raw: bytes) -> Tuple[Dict[str, List[str]], int]:
    """Parse a PCD header, returning the keyword->values map and the byte offset of the body."""
    header: Dict[str, List[str]] = {}
    header_size = 0
    for raw_line in raw.splitlines(keepends=True):
        header_size += len(raw_line)
        line = raw_line.decode("ascii", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0].lower()] = parts[1:]
        if parts[0].lower() == "data":
            break
    return header, header_size


def _pcd_field_dtypes(header: Dict[str, List[str]]) -> Dict[str, np.dtype]:
    """Build a per-field numpy dtype map from a parsed PCD header."""
    fields = header["fields"]
    sizes = [int(v) for v in header["size"]]
    types = header["type"]
    field_dtypes: Dict[str, np.dtype] = {}
    for name, size, type_char in zip(fields, sizes, types):
        field_dtypes[name] = np.dtype(_PCD_TYPE_MAP[(size, type_char.upper())])
    return field_dtypes


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Decompress a PCL ``binary_compressed`` payload (LZF / liblzf format).

    NOTE: This is a pure-python implementation (no extra dependencies). It is adequate for
    read-time access; the back-reference copy is necessarily byte-wise because matches may overlap.
    """
    output = bytearray()
    i = 0
    data_len = len(data)
    while i < data_len:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            # Literal run of (ctrl + 1) bytes.
            length = ctrl + 1
            output.extend(data[i : i + length])
            i += length
        else:
            # Back reference.
            length = ctrl >> 5
            ref_offset = (ctrl & 0x1F) << 8
            if length == 7:
                length += data[i]
                i += 1
            ref_offset += data[i]
            i += 1
            length += 2
            ref_pos = len(output) - ref_offset - 1
            if ref_pos < 0:
                raise ValueError("Invalid LZF back reference")
            for _ in range(length):
                output.append(output[ref_pos])
                ref_pos += 1
    if len(output) != expected_size:
        raise ValueError(f"LZF output size {len(output)} != expected {expected_size}")
    return bytes(output)


def _read_pcd_fields(pcd_path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """Read all fields of a PCD file into a name->array dict, supporting binary_compressed/binary.

    :return: A tuple of (per-field arrays, number of points).
    """
    raw = pcd_path.read_bytes()
    header, header_size = _parse_pcd_header(raw)
    data_format = header["data"][0].lower() if header.get("data") else "binary"

    fields: List[str] = header["fields"]
    counts = [int(v) for v in header.get("count", ["1"] * len(fields))]
    field_dtypes = _pcd_field_dtypes(header)
    num_points = int(header["points"][0]) if "points" in header else int(header["width"][0])

    field_arrays: Dict[str, np.ndarray] = {}
    if data_format == "binary_compressed":
        # Layout after the header: <compressed_size:u4><uncompressed_size:u4><lzf payload>.
        # The decompressed payload is structure-of-arrays: each field stored contiguously.
        compressed_size, uncompressed_size = struct.unpack("<II", raw[header_size : header_size + 8])
        body = _lzf_decompress(raw[header_size + 8 : header_size + 8 + compressed_size], uncompressed_size)

        field_byte_sizes = [field_dtypes[name].itemsize * count for name, count in zip(fields, counts)]
        offsets = np.cumsum([0] + [size * num_points for size in field_byte_sizes])
        for idx, name in enumerate(fields):
            field_arrays[name] = np.frombuffer(
                body[int(offsets[idx]) : int(offsets[idx + 1])], dtype=field_dtypes[name], count=num_points
            )
    elif data_format == "binary":
        # Array-of-structures: one interleaved record per point.
        struct_dtype = np.dtype([(name, field_dtypes[name]) for name in fields])
        records = np.frombuffer(raw[header_size:], dtype=struct_dtype, count=num_points)
        for name in fields:
            field_arrays[name] = records[name]
    else:
        raise NotImplementedError(f"Unsupported nuReasoning PCD DATA format: {data_format!r}")

    return field_arrays, num_points


def load_nureasoning_point_cloud_data_from_path(pcd_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load a nuReasoning lidar point cloud from a ``.pcd`` file.

    :param pcd_path: Path to the ``.pcd`` file.
    :return: A tuple of (xyz point cloud of shape ``(N, 3)``, feature dict keyed by serialized
        :class:`LidarFeature`).
    """
    assert pcd_path.exists(), f"Lidar file not found: {pcd_path}"

    fields, num_points = _read_pcd_fields(pcd_path)

    # Map the per-point lidar source (lidar_info) to py123d LidarIDs.
    lidar_ids = np.zeros(num_points, dtype=np.uint8)
    lidar_info = fields["lidar_info"]
    for nureasoning_lidar_id, lidar_id in NUREASONING_LIDAR_DICT.items():
        lidar_ids[lidar_info == nureasoning_lidar_id] = int(lidar_id)

    point_cloud_3d = np.column_stack((fields["x"], fields["y"], fields["z"])).astype(np.float32)
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): fields["intensity"].astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): fields["ring"].astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }

    return point_cloud_3d, point_cloud_features
