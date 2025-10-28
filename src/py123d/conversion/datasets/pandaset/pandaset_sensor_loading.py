from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from py123d.conversion.datasets.pandaset.pandaset_utlis import read_pkl_gz
from py123d.conversion.utils.sensor_utils.lidar_index_registry import PandasetLidarIndex
from py123d.datatypes.sensors.lidar.lidar import LiDARType
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.transform.transform_se3 import convert_absolute_to_relative_points_3d_array


def load_pandaset_global_lidar_pc_from_path(pkl_gz_path: Union[Path, str]) -> Dict[LiDARType, np.ndarray]:
    # NOTE: The Pandaset dataset stores both front and top LiDAR data in the same gzip-pickle file.
    # We need to separate them based on the laser_number field.
    # See here: https://github.com/scaleapi/pandaset-devkit/blob/master/python/pandaset/sensors.py#L160

    all_lidar_df = read_pkl_gz(pkl_gz_path)
    top_lidar_df: pd.DataFrame = all_lidar_df[all_lidar_df["d"] == 0]
    front_lidar_df: pd.DataFrame = all_lidar_df[all_lidar_df["d"] == 1]

    # Remove the 't' (timestamp) and 'd' (laser id) columns
    top_lidar_df = top_lidar_df.drop(columns=["t", "d"])
    front_lidar_df = front_lidar_df.drop(columns=["t", "d"])

    return {LiDARType.LIDAR_TOP: top_lidar_df.to_numpy(), LiDARType.LIDAR_FRONT: front_lidar_df.to_numpy()}


def load_pandaset_lidars_pc_from_path(
    pkl_gz_path: Union[Path, str], ego_state_se3: EgoStateSE3
) -> Dict[LiDARType, np.ndarray]:

    lidar_pc_dict = load_pandaset_global_lidar_pc_from_path(pkl_gz_path)

    for lidar_type in lidar_pc_dict.keys():
        lidar_pc_dict[lidar_type][..., PandasetLidarIndex.XYZ] = convert_absolute_to_relative_points_3d_array(
            ego_state_se3.rear_axle_se3,
            lidar_pc_dict[lidar_type][..., PandasetLidarIndex.XYZ],
        )

        # relative_points = (points_3d_array - t_origin) @ R_origin

    # Pass the loaded point clouds to the appropriate LiDAR types
    return lidar_pc_dict
