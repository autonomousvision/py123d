from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from py123d.datatypes import LidarFeature, LidarID, LidarMetadata
from py123d.geometry import PoseSE3
from py123d.geometry.transform import reframe_points_3d_array

# nuScenes-panoptic encodes each point as ``category_idx * PANOPTIC_DIVISOR + instance_id``; the
# instance id is the remainder (``0`` for "stuff" / un-instanced points). See
# ``nuscenes/utils/data_io.py::panoptic_to_lidarseg``.
NUSCENES_PANOPTIC_DIVISOR: int = 1000


def load_nuscenes_point_cloud_data_from_path(
    pcd_path: Path,
    lidar_metadatas: Dict[LidarID, LidarMetadata],
    lidarseg_path: Optional[Union[Path, str]] = None,
    panoptic_path: Optional[Union[Path, str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads nuScenes Lidar point clouds from the original binary files.

    Per-point segmentation is attached from two complementary, keyframe-only add-ons (each optional, so
    sparse): the SEMANTIC class id comes from the **lidarseg** ``.bin`` and the INSTANCE id from the
    **panoptic** ``.npz``. The semantic id is intentionally *not* derived from ``panoptic // 1000`` —
    panoptic demotes a few percent of thing-class points (car/ped/motorcycle) outside a valid instance
    to noise, whereas lidarseg is the canonical, lossless semantic ground truth used in the literature.
    """

    lidar_data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)  # Indices: x, y, z, intensity, ring
    assert lidar_data.ndim == 2 and lidar_data.shape[1] == 5, (
        f"Expected Lidar data to have shape (N, 5) for nuScenes, but got shape {lidar_data.shape}."
    )
    lidar_extrinsic = lidar_metadatas[LidarID.LIDAR_TOP].lidar_to_imu_se3

    lidar_ids = np.zeros(lidar_data.shape[0], dtype=np.uint8)  # nuScenes only has a top lidar.
    lidar_ids[:] = int(LidarID.LIDAR_TOP)

    # convert lidar to ego frame
    point_cloud_3d = reframe_points_3d_array(
        from_origin=lidar_extrinsic,
        to_origin=PoseSE3.identity(),
        points_3d_array=lidar_data[..., :3],  # type: ignore
    )
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): lidar_data[..., 3].astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): lidar_data[..., 4].astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }

    # Per-point segmentation, available for nuScenes keyframes only (sparse): semantic from lidarseg,
    # instance from panoptic. Each is attached independently when its source file is present.
    semantic = _load_nuscenes_lidarseg(lidarseg_path, num_points=lidar_data.shape[0])
    if semantic is not None:
        point_cloud_features[LidarFeature.SEMANTIC.serialize()] = semantic
    instance = _load_nuscenes_panoptic_instance(panoptic_path, num_points=lidar_data.shape[0])
    if instance is not None:
        point_cloud_features[LidarFeature.INSTANCE.serialize()] = instance

    return point_cloud_3d.astype(np.float32), point_cloud_features


def _load_nuscenes_lidarseg(lidarseg_path: Optional[Union[Path, str]], num_points: int) -> Optional[np.ndarray]:
    """Loads nuScenes per-point SEMANTIC class ids for one keyframe, or ``None`` if unavailable.

    nuScenes-lidarseg stores one ``uint8`` label per point in ``lidarseg/<version>/<sd_token>_lidarseg.bin``,
    row-aligned 1:1 (same order) with the sibling ``.pcd.bin`` point cloud. Raw ids are ``0..31`` (see
    :class:`~py123d.parser.lidar_segmentation_registry.NuScenesLidarSegmentationLabel`). This is the
    canonical semantic ground truth (used directly in the nuScenes-lidarseg benchmark and literature).

    :param lidarseg_path: Absolute path to the ``_lidarseg.bin`` file, or ``None`` if there is none.
    :param num_points: Expected number of points, asserted to equal the number of labels.
    :return: An ``(N,)`` uint8 array of semantic class ids, or ``None`` if unavailable.
    """
    semantic: Optional[np.ndarray] = None
    if lidarseg_path is not None and Path(lidarseg_path).exists():
        semantic = np.fromfile(lidarseg_path, dtype=np.uint8)
        assert semantic.shape[0] == num_points, (
            f"nuScenes lidarseg labels are misaligned with the point cloud "
            f"({semantic.shape[0]} labels vs {num_points} points) at {lidarseg_path}."
        )
    return semantic


def _load_nuscenes_panoptic_instance(
    panoptic_path: Optional[Union[Path, str]], num_points: int
) -> Optional[np.ndarray]:
    """Loads nuScenes per-point INSTANCE ids for one keyframe, or ``None`` if unavailable.

    nuScenes-panoptic stores one label per point in ``panoptic/<version>/<sd_token>_panoptic.npz``
    (key ``"data"``), row-aligned 1:1 with the sibling ``.pcd.bin``. Each label is
    ``category_idx * NUSCENES_PANOPTIC_DIVISOR + instance_id``; we keep only the instance id (the
    remainder), cast to uint16 (``0`` for un-instanced "stuff" points). The semantic component is
    deliberately ignored here — see :func:`_load_nuscenes_lidarseg`.

    :param panoptic_path: Absolute path to the ``_panoptic.npz`` file, or ``None`` if there is none.
    :param num_points: Expected number of points, asserted to equal the number of labels.
    :return: An ``(N,)`` uint16 array of instance ids, or ``None`` if unavailable.
    """
    instance: Optional[np.ndarray] = None
    if panoptic_path is not None and Path(panoptic_path).exists():
        panoptic = np.load(panoptic_path)["data"]
        assert panoptic.shape[0] == num_points, (
            f"nuScenes panoptic labels are misaligned with the point cloud "
            f"({panoptic.shape[0]} labels vs {num_points} points) at {panoptic_path}."
        )
        instance = (panoptic % NUSCENES_PANOPTIC_DIVISOR).astype(np.uint16)
    return instance
