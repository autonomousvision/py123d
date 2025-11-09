from pathlib import Path
from typing import Dict

import numpy as np

from py123d.conversion.registry.lidar_index_registry import NuScenesLiDARIndex
from py123d.datatypes.metadata import LogMetadata
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.geometry.pose import PoseSE3
from py123d.geometry.transform.transform_se3 import convert_points_3d_array_between_origins


def load_nuscenes_lidar_pcs_from_file(pcd_path: Path, log_metadata: LogMetadata) -> Dict[LiDARType, np.ndarray]:
    lidar_pc = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, len(NuScenesLiDARIndex))

    # convert lidar to ego frame
    lidar_extrinsic = log_metadata.lidar_metadata[LiDARType.LIDAR_TOP].extrinsic
    lidar_pc[..., NuScenesLiDARIndex.XYZ] = convert_points_3d_array_between_origins(
        from_origin=lidar_extrinsic,
        to_origin=PoseSE3(0, 0, 0, 1.0, 0, 0, 0),
        points_3d_array=lidar_pc[..., NuScenesLiDARIndex.XYZ],
    )
    return {LiDARType.LIDAR_TOP: lidar_pc}
