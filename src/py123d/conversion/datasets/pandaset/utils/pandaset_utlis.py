import gzip
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from py123d.geometry import StateSE3, Vector3D
from py123d.geometry.transform.transform_se3 import translate_se3_along_body_frame


def read_json(json_file: Path):
    """Helper function to read a json file as dict."""
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def read_pkl_gz(pkl_gz_file: Path):
    """Helper function to read a pkl.gz file as dict."""
    with gzip.open(pkl_gz_file, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def pandaset_pose_dict_to_state_se3(pose_dict: Dict[str, Dict[str, float]]) -> StateSE3:
    """Helper function for pandaset to convert a pose dict to StateSE3.

    :param pose_dict: The input pose dict.
    :return: The converted StateSE3.
    """
    return StateSE3(
        x=pose_dict["position"]["x"],
        y=pose_dict["position"]["y"],
        z=pose_dict["position"]["z"],
        qw=pose_dict["heading"]["w"],
        qx=pose_dict["heading"]["x"],
        qy=pose_dict["heading"]["y"],
        qz=pose_dict["heading"]["z"],
    )


def rotate_pandaset_pose_to_iso_coordinates(pose: StateSE3) -> StateSE3:
    """Helper function for pandaset to rotate a pose to ISO coordinate system (x: forward, y: left, z: up).

    NOTE: Pandaset uses a different coordinate system (x: right, y: forward, z: up).
    [1] https://arxiv.org/pdf/2112.12610.pdf

    :param pose: The input pose.
    :return: The rotated pose.
    """
    F = np.array(
        [
            [0.0, 1.0, 0.0],  # new X = old Y (forward)
            [-1.0, 0.0, 0.0],  # new Y = old -X (left)
            [0.0, 0.0, 1.0],  # new Z = old Z (up)
        ],
        dtype=np.float64,
    ).T
    transformation_matrix = pose.transformation_matrix.copy()
    transformation_matrix[0:3, 0:3] = transformation_matrix[0:3, 0:3] @ F

    return StateSE3.from_transformation_matrix(transformation_matrix)


def main_lidar_to_rear_axle(pose: StateSE3) -> StateSE3:

    F = np.array(
        [
            [0.0, 1.0, 0.0],  # new X = old Y (forward)
            [-1.0, 0.0, 0.0],  # new Y = old X (left)
            [0.0, 0.0, 1.0],  # new Z = old Z (up)
        ],
        dtype=np.float64,
    ).T
    transformation_matrix = pose.transformation_matrix.copy()
    transformation_matrix[0:3, 0:3] = transformation_matrix[0:3, 0:3] @ F

    rotated_pose = StateSE3.from_transformation_matrix(transformation_matrix)

    imu_pose = translate_se3_along_body_frame(rotated_pose, vector_3d=Vector3D(x=-0.840, y=0.0, z=0.0))

    return imu_pose
