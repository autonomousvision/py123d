from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from py123d.conversion.datasets.pandaset.utils.pandaset_utlis import (
    global_main_lidar_to_global_imu,
    pandaset_pose_dict_to_pose_se3,
    read_json,
    read_pkl_gz,
)
from py123d.conversion.registry.lidar_index_registry import PandasetLiDARIndex
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.geometry.transform.transform_se3 import convert_absolute_to_relative_points_3d_array


def load_pandaset_global_lidar_pc_from_path(pkl_gz_path: Union[Path, str]) -> Dict[LiDARType, np.ndarray]:
    """Loads Pandaset LiDAR point clouds from a gzip-pickle file (pickled pandas DataFrame)."""

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


def load_pandaset_lidars_pcs_from_file(
    pkl_gz_path: Union[Path, str],
    iteration: Optional[int],
) -> Dict[LiDARType, np.ndarray]:
    """Loads Pandaset LiDAR point clouds from a gzip-pickle file and converts them to ego frame."""

    pkl_gz_path = Path(pkl_gz_path)
    assert pkl_gz_path.exists(), f"Pandaset LiDAR file not found: {pkl_gz_path}"
    lidar_pc_dict = load_pandaset_global_lidar_pc_from_path(pkl_gz_path)
    ego_pose = global_main_lidar_to_global_imu(
        pandaset_pose_dict_to_pose_se3(read_json(pkl_gz_path.parent / "poses.json")[iteration])
    )
    for lidar_type in lidar_pc_dict.keys():
        lidar_pc_dict[lidar_type][..., PandasetLiDARIndex.XYZ] = convert_absolute_to_relative_points_3d_array(
            ego_pose,
            lidar_pc_dict[lidar_type][..., PandasetLiDARIndex.XYZ],
        )

    return lidar_pc_dict
