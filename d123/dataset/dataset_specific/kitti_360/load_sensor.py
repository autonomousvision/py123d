from pathlib import Path

import numpy as np

from d123.common.datatypes.sensor.lidar import LiDAR, LiDARMetadata


def load_kitti360_lidar_from_path(filepath: Path, lidar_metadata: LiDARMetadata) -> LiDAR:
    assert filepath.exists(), f"LiDAR file not found: {filepath}"
    pcd = np.fromfile(filepath, dtype=np.float32)
    pcd = np.reshape(pcd,[-1,4]) # [N,4]

    xyz = pcd[:, :3] 
    intensity = pcd[:, 3]    

    ones = np.ones((xyz.shape[0], 1), dtype=pcd.dtype)
    points_h = np.concatenate([xyz, ones], axis=1)  #[N,4]

    transformed_h = lidar_metadata.extrinsic @ points_h.T   #[4,N]

    transformed_xyz = transformed_h[:3, :]      # (3,N)

    intensity_row = intensity[np.newaxis, :]    # (1,N)

    point_cloud_4xN = np.vstack([transformed_xyz, intensity_row]).astype(np.float32)  # (4,N)

    return LiDAR(metadata=lidar_metadata, point_cloud=point_cloud_4xN)
