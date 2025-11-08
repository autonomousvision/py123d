import logging
from pathlib import Path
from typing import Dict

import numpy as np

from py123d.conversion.registry.lidar_index_registry import Kitti360LiDARIndex
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.geometry.pose import PoseSE3
from py123d.geometry.transform.transform_se3 import convert_points_3d_array_between_origins


def load_kitti360_lidar_pcs_from_file(filepath: Path, log_metadata: LogMetadata) -> Dict[LiDARType, np.ndarray]:
    if not filepath.exists():
        logging.warning(f"LiDAR file does not exist: {filepath}. Returning empty point cloud.")
        return {LiDARType.LIDAR_TOP: np.zeros((1, len(Kitti360LiDARIndex)), dtype=np.float32)}

    lidar_extrinsic = log_metadata.lidar_metadata[LiDARType.LIDAR_TOP].extrinsic
    lidar_pc = np.fromfile(filepath, dtype=np.float32)
    lidar_pc = np.reshape(lidar_pc, [-1, len(Kitti360LiDARIndex)])

    lidar_pc[..., Kitti360LiDARIndex.XYZ] = convert_points_3d_array_between_origins(
        from_origin=lidar_extrinsic,
        to_origin=PoseSE3(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        points_3d_array=lidar_pc[..., Kitti360LiDARIndex.XYZ],
    )

    return {LiDARType.LIDAR_TOP: lidar_pc}
