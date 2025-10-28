from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from py123d.datatypes.sensors.lidar.lidar import LiDARType


def load_av2_sensor_lidar_pc_from_path(feather_path: Union[Path, str]) -> Dict[LiDARType, np.ndarray]:
    # NOTE: The AV2 dataset stores both top and down LiDAR data in the same feather file.
    # We need to separate them based on the laser_number field.
    # See here: https://github.com/argoverse/av2-api/issues/77#issuecomment-1178040867
    all_lidar_df = pd.read_feather(feather_path, columns=["x", "y", "z", "intensity", "laser_number"])

    lidar_down = all_lidar_df[all_lidar_df["laser_number"] >= 32]
    lidar_top = all_lidar_df[all_lidar_df["laser_number"] < 32]

    lidar_down = lidar_down.drop(columns=["laser_number"])
    lidar_top = lidar_top.drop(columns=["laser_number"])

    return {LiDARType.LIDAR_DOWN: lidar_down.to_numpy(), LiDARType.LIDAR_TOP: lidar_top.to_numpy()}
