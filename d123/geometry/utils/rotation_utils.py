from typing import Union

import numpy as np
import numpy.typing as npt
import pyquaternion

from d123.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex


def normalize_angle(angle: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float or array of floats
    :return: normalized angle or array of normalized angles
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def get_rotation_matrices_from_euler_array(euler_angles_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert Euler angles to rotation matrices using Tait-Bryan ZYX convention (yaw-pitch-roll).

    Convention: Intrinsic rotations in order Z-Y-X (yaw, pitch, roll)
    Equivalent to: R = R_x(roll) @ R_y(pitch) @ R_z(yaw)
    """
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

    # ZYX Tait-Bryan rotation matrix elements
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
    return get_rotation_matrices_from_euler_array(euler_angles[None, ...])[0]


def get_rotation_matrices_from_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert quaternion_array.ndim == 2 and quaternion_array.shape[1] == len(QuaternionIndex)
    # TODO: Optimize this function to avoid the for loop, possibly by using pyquaternion's internal methods directly.
    rotation_matrices = np.zeros((quaternion_array.shape[0], 3, 3), dtype=np.float64)
    for i, quaternion in enumerate(quaternion_array):
        rotation_matrices[i] = pyquaternion.Quaternion(array=quaternion).rotation_matrix
    return rotation_matrices


def get_rotation_matrix_from_quaternion_array(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert quaternion.ndim == 1 and quaternion.shape[0] == len(QuaternionIndex)
    return get_rotation_matrices_from_quaternion_array(quaternion[None, :])[0]


def conjugate_quaternion_array(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the conjugate of an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of conjugated quaternions.
    """
    assert quaternion.ndim >= 1
    assert quaternion.shape[-1] == len(QuaternionIndex)
    conjugated_quaternions = np.zeros_like(quaternion)
    conjugated_quaternions[..., QuaternionIndex.SCALAR] = quaternion[..., QuaternionIndex.SCALAR]
    conjugated_quaternions[..., QuaternionIndex.VECTOR] = -quaternion[..., QuaternionIndex.VECTOR]
    return conjugated_quaternions


def invert_quaternion_array(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the inverse of an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of inverted quaternions.
    """
    assert quaternion.ndim >= 1
    assert quaternion.shape[-1] == len(QuaternionIndex)
    norm_squared = np.sum(quaternion**2, axis=-1, keepdims=True)
    assert np.all(norm_squared > 0), "Cannot invert a quaternion with zero norm."
    conjugated_quaternions = conjugate_quaternion_array(quaternion)
    inverted_quaternions = conjugated_quaternions / norm_squared
    return inverted_quaternions


def normalize_quaternion_array(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalizes an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of normalized quaternions.
    """
    assert quaternion.ndim >= 1
    assert quaternion.shape[-1] == len(QuaternionIndex)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    assert np.all(norm > 0), "Cannot normalize a quaternion with zero norm."
    normalized_quaternions = quaternion / norm
    return normalized_quaternions


def multiply_quaternion_arrays(q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiplies two arrays of quaternions element-wise.
    in the order [qw, qx, qy, qz].
    :param q1: First array of quaternions.
    :param q2: Second array of quaternions.
    :return: Array of resulting quaternions after multiplication.
    """
    assert q1.ndim >= 1
    assert q2.ndim >= 1
    assert q1.shape[-1] == q2.shape[-1] == len(QuaternionIndex)

    # Vectorized quaternion multiplication
    qw1, qx1, qy1, qz1 = (
        q1[..., QuaternionIndex.QW],
        q1[..., QuaternionIndex.QX],
        q1[..., QuaternionIndex.QY],
        q1[..., QuaternionIndex.QZ],
    )
    qw2, qx2, qy2, qz2 = (
        q2[..., QuaternionIndex.QW],
        q2[..., QuaternionIndex.QX],
        q2[..., QuaternionIndex.QY],
        q2[..., QuaternionIndex.QZ],
    )

    quaternions = np.stack(
        [
            qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2,
            qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2,
            qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2,
            qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2,
        ],
        axis=-1,
    )
    return quaternions
