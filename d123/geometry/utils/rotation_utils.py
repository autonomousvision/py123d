from typing import Union

import numpy as np
import numpy.typing as npt

from d123.geometry.geometry_index import EulerAnglesIndex

# TODO: move this somewhere else
# TODO: Maybe rename wrap angle?
# TODO: Add implementation for torch, jax, or whatever else is needed.


def normalize_angle(angle: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float or array of floats
    :return: normalized angle or array of normalized angles
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def get_rotation_matrices_from_euler_array(euler_angles_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert euler_angles_array.ndim == 2 and euler_angles_array.shape[1] == len(EulerAnglesIndex)

    # Extract roll, pitch, yaw for all samples at once
    roll = euler_angles_array[:, EulerAnglesIndex.ROLL]
    pitch = euler_angles_array[:, EulerAnglesIndex.PITCH]
    yaw = euler_angles_array[:, EulerAnglesIndex.YAW]

    # Compute sin/cos for all angles at once
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    # Build rotation matrices for entire batch
    batch_size = euler_angles_array.shape[0]
    rotation_matrices = np.zeros((batch_size, 3, 3), dtype=np.float64)

    # R_x @ R_y @ R_z components
    rotation_matrices[:, 0, 0] = cos_pitch * cos_yaw
    rotation_matrices[:, 0, 1] = -cos_pitch * sin_yaw
    rotation_matrices[:, 0, 2] = sin_pitch

    rotation_matrices[:, 1, 0] = sin_roll * sin_pitch * cos_yaw + cos_roll * sin_yaw
    rotation_matrices[:, 1, 1] = -sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw
    rotation_matrices[:, 1, 2] = -sin_roll * cos_pitch

    rotation_matrices[:, 2, 0] = -cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw
    rotation_matrices[:, 2, 1] = cos_roll * sin_pitch * sin_yaw + sin_roll * cos_yaw
    rotation_matrices[:, 2, 2] = cos_roll * cos_pitch

    return rotation_matrices


def get_rotation_matrix_from_euler_array(euler_angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert euler_angles.ndim == 1 and euler_angles.shape[0] == len(EulerAnglesIndex)
    return get_rotation_matrices_from_euler_array(euler_angles[None, :])[0]
