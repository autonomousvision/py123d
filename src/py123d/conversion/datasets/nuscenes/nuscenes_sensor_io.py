from pathlib import Path
from typing import Dict

import numpy as np

from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.lidar.lidar import LiDARType
from py123d.datatypes.sensors.lidar.lidar_index import NuScenesLidarIndex
from py123d.geometry.se import StateSE3
from py123d.geometry.transform.transform_se3 import convert_points_3d_array_between_origins


def load_nuscenes_lidar_pcs_from_file(pcd_path: Path, log_metadata: LogMetadata) -> Dict[LiDARType, np.ndarray]:
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)

    # convert lidar to ego frame
    lidar_extrinsic = log_metadata.lidar_metadata[LiDARType.LIDAR_TOP].extrinsic

    points[..., NuScenesLidarIndex.XYZ] = convert_points_3d_array_between_origins(
        from_origin=lidar_extrinsic,
        to_origin=StateSE3(0, 0, 0, 1.0, 0, 0, 0),
        points_3d_array=points[..., NuScenesLidarIndex.XYZ],
    )
    lidar_pcs_dict: Dict[LiDARType, np.ndarray] = {LiDARType.LIDAR_TOP: points}

    return lidar_pcs_dict
