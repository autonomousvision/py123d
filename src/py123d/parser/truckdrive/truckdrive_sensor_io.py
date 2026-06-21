"""TruckDrive on-disk point cloud loaders for py123d path-based lidar I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from py123d.datatypes import LidarFeature, LidarID, LidarMetadata

_AEVA_JOINT_PATH_MARKER = "lidar/aeva/joint_lidars/points"
_OUSTER_PATH_MARKER = "lidar/ouster/"


def _load_aeva_joint_bin(bin_path: Path, lidar_id: LidarID) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load an Aeva joint-lidar ``.bin`` (float64, 11 columns per point)."""
    raw = np.fromfile(bin_path, dtype=np.float64)
    if raw.size == 0:
        point_count = 0
        points = raw.reshape((0, 11))
    else:
        if raw.size % 11 != 0:
            raise ValueError(f"{bin_path} is not divisible by 11 Aeva columns")
        points = raw.reshape((-1, 11))
        point_count = points.shape[0]

    point_cloud_3d = points[:, :3].astype(np.float32, copy=False)
    ids = np.full(point_count, int(lidar_id), dtype=np.uint8)
    features: Dict[str, np.ndarray] = {
        LidarFeature.IDS.serialize(): ids,
    }
    if point_count > 0:
        features[LidarFeature.INTENSITY.serialize()] = np.clip(points[:, 3], 0, 255).astype(np.uint8)
        features[LidarFeature.TIMESTAMPS.serialize()] = points[:, 6].astype(np.int64)
    return point_cloud_3d, features


def _load_ouster_bin(bin_path: Path, lidar_id: LidarID) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load an Ouster ``.bin`` (float32, 7 columns per point)."""
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size == 0:
        point_count = 0
        points = raw.reshape((0, 7))
    else:
        if raw.size % 7 != 0:
            raise ValueError(f"{bin_path} is not divisible by 7 Ouster columns")
        points = raw.reshape((-1, 7))
        point_count = points.shape[0]

    point_cloud_3d = points[:, :3].astype(np.float32, copy=False)
    ids = np.full(point_count, int(lidar_id), dtype=np.uint8)
    features: Dict[str, np.ndarray] = {
        LidarFeature.IDS.serialize(): ids,
    }
    if point_count > 0:
        features[LidarFeature.INTENSITY.serialize()] = np.clip(points[:, 3], 0, 255).astype(np.uint8)
        features[LidarFeature.TIMESTAMPS.serialize()] = points[:, 4].astype(np.int64)
        features[LidarFeature.CHANNEL.serialize()] = points[:, 6].astype(np.uint8)
    return point_cloud_3d, features


def _resolve_ouster_lidar_id(bin_path: Path) -> LidarID:
    path_str = bin_path.as_posix()
    if "forward_center" in path_str:
        return LidarID.LIDAR_FRONT
    if "sideward_left" in path_str:
        return LidarID.LIDAR_SIDE_LEFT
    if "sideward_right" in path_str:
        return LidarID.LIDAR_SIDE_RIGHT
    raise ValueError(f"Could not resolve Ouster lidar id from path: {bin_path}")


def load_truckdrive_point_cloud_data_from_path(
    bin_path: Union[Path, str],
    lidar_metadatas: Optional[Dict[LidarID, LidarMetadata]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load a TruckDrive lidar ``.bin`` into py123d point cloud layout.

    Points are already expressed in the native sensor frame on disk; no extrinsic
  reframe is applied here.

    :param bin_path: Absolute path to the ``.bin`` file.
    :param lidar_metadatas: Optional per-lidar metadata (used to pick ``LidarID`` for Aeva merged).
    :return: ``(point_cloud_3d, point_cloud_features)`` with xyz float32 and feature arrays.
    """
    bin_path = Path(bin_path)
    path_str = bin_path.as_posix()

    if _AEVA_JOINT_PATH_MARKER in path_str:
        lidar_id = LidarID.LIDAR_MERGED
        if lidar_metadatas is not None and len(lidar_metadatas) == 1:
            lidar_id = next(iter(lidar_metadatas))
        return _load_aeva_joint_bin(bin_path, lidar_id)

    if _OUSTER_PATH_MARKER in path_str:
        return _load_ouster_bin(bin_path, _resolve_ouster_lidar_id(bin_path))

    raise ValueError(f"Unrecognized TruckDrive lidar path: {bin_path}")
