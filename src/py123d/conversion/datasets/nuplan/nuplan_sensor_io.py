import io
from pathlib import Path
from typing import Dict

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.datasets.nuplan.utils.nuplan_constants import NUPLAN_LIDAR_DICT
from py123d.conversion.registry import NuPlanLiDARIndex
from py123d.datatypes.sensors import LiDARType

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud


def load_nuplan_lidar_pcs_from_file(pcd_path: Path) -> Dict[LiDARType, np.ndarray]:
    """Loads nuPlan LiDAR point clouds from a ``.pcd`` file."""

    assert pcd_path.exists(), f"LiDAR file not found: {pcd_path}"
    with open(pcd_path, "rb") as fp:
        buffer = io.BytesIO(fp.read())

    merged_lidar_pc = LidarPointCloud.from_buffer(buffer, "pcd").points

    lidar_pcs_dict: Dict[LiDARType, np.ndarray] = {}
    for lidar_id, lidar_type in NUPLAN_LIDAR_DICT.items():
        mask = merged_lidar_pc[-1, :] == lidar_id
        lidar_pc = merged_lidar_pc[: len(NuPlanLiDARIndex), mask].T.astype(np.float32)
        lidar_pcs_dict[lidar_type] = lidar_pc

    return lidar_pcs_dict
