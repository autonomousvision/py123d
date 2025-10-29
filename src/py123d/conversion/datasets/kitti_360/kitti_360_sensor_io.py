from pathlib import Path

from typing import Dict
import numpy as np
import logging
from py123d.datatypes.sensors.lidar.lidar import LiDAR, LiDARMetadata, LiDARType
from py123d.conversion.datasets.kitti_360.kitti_360_helper import get_lidar_extrinsic

def load_kitti360_lidar_pcs_from_file(filepath: Path) -> Dict[LiDARType, np.ndarray]:
    if not filepath.exists():
        logging.warning(f"LiDAR file does not exist: {filepath}. Returning empty point cloud.")
        return {LiDARType.LIDAR_TOP: np.zeros((1, 4), dtype=np.float32)}
    
    pcd = np.fromfile(filepath, dtype=np.float32)
    pcd = np.reshape(pcd,[-1,4]) # [N,4]

    xyz = pcd[:, :3] 
    intensity = pcd[:, 3]    

    ones = np.ones((xyz.shape[0], 1), dtype=pcd.dtype)
    points_h = np.concatenate([xyz, ones], axis=1)  #[N,4]

    transformed_h = get_lidar_extrinsic() @ points_h.T   #[4,N]
    # transformed_h = lidar_metadata.extrinsic.transformation_matrix @ points_h.T   #[4,N]

    transformed_xyz = transformed_h[:3, :]      # (3,N)

    intensity_row = intensity[np.newaxis, :]    # (1,N)

    point_cloud_4xN = np.vstack([transformed_xyz, intensity_row]).astype(np.float32)  # (4,N)

    point_cloud_Nx4 = point_cloud_4xN.T  # (N,4)

    return {LiDARType.LIDAR_TOP: point_cloud_Nx4}
