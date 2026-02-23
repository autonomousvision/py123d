import io
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.datasets.nuplan.utils.nuplan_constants import NUPLAN_LIDAR_DICT
from py123d.datatypes.sensors import LidarFeature

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud


def load_nuplan_point_cloud_data_from_path(pcd_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads nuPlan Lidar point clouds from a ``.pcd`` file."""

    assert pcd_path.exists(), f"Lidar file not found: {pcd_path}"
    with open(pcd_path, "rb") as fp:
        buffer = io.BytesIO(fp.read())

    # Indices: x, y, z, intensity, ring, lidar_id
    merged_lidar_pc = LidarPointCloud.from_buffer(buffer, "pcd").points
    lidar_ids = np.zeros(merged_lidar_pc.shape[1], dtype=np.uint8)

    for nuplan_lidar_id, lidar_id in NUPLAN_LIDAR_DICT.items():
        mask = merged_lidar_pc[-1, :] == nuplan_lidar_id
        lidar_ids[mask] = int(lidar_id)

    point_cloud_3d = merged_lidar_pc[:3, :].T.astype(np.float32)
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): merged_lidar_pc[3, :].astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): merged_lidar_pc[4, :].astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }

    return point_cloud_3d, point_cloud_features
