import numpy as np
from pathlib import Path
from typing import Dict
from py123d.datatypes.sensors.lidar.lidar import LiDARType

def load_nuscenes_lidar_pcs_from_file(pcd_path: Path) -> Dict[LiDARType, np.ndarray]:
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
    
    lidar_pcs_dict: Dict[LiDARType, np.ndarray] = {
        LiDARType.LIDAR_TOP: points
    }
    
    return lidar_pcs_dict
