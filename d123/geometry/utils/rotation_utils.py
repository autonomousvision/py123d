from typing import Union

import numpy as np
import numpy.typing as npt

from d123.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex

# import pyquaternion


def batch_matmul(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Batch matrix multiplication for arrays of matrices.
    # TODO: move somewhere else

    :param A: Array of shape (..., M, N)
    :param B: Array of shape (..., N, P)
    :return: Array of shape (..., M, P) resulting from batch matrix multiplication of A and B.
    """
    assert A.ndim >= 2 and B.ndim >= 2
    assert (
        A.shape[-1] == B.shape[-2]
    ), f"Inner dimensions must match for matrix multiplication, got {A.shape} and {B.shape}"
    return np.einsum("...ij,...jk->...ik", A, B)


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
    assert euler_angles_array.ndim >= 1 and euler_angles_array.shape[-1] == len(EulerAnglesIndex)

    # Store original shape for reshaping later
    original_shape = euler_angles_array.shape[:-1]

    # Flatten to 2D if needed
    if euler_angles_array.ndim > 2:
        euler_angles_array_ = euler_angles_array.reshape(-1, len(EulerAnglesIndex))
    else:
        euler_angles_array_ = euler_angles_array

    # Extract roll, pitch, yaw for all samples at once
    roll = euler_angles_array_[:, EulerAnglesIndex.ROLL]
    pitch = euler_angles_array_[:, EulerAnglesIndex.PITCH]
    yaw = euler_angles_array_[:, EulerAnglesIndex.YAW]

    # Compute sin/cos for all angles at once
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    # Build rotation matrices for entire batch
    batch_size = euler_angles_array_.shape[0]
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

    # Reshape back to original batch dimensions + (3, 3)
    if len(original_shape) > 1:
        rotation_matrices = rotation_matrices.reshape(original_shape + (3, 3))

    return rotation_matrices


def get_euler_array_from_rotation_matrix(rotation_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    raise NotImplementedError


def get_quaternion_array_from_rotation_matrices(rotation_matrices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert rotation_matrices.ndim == 3
    assert rotation_matrices.shape[-1] == rotation_matrices.shape[-2] == 3
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    # TODO: Update with:
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

    N = rotation_matrices.shape[0]
    quaternions = np.zeros((N, 4), dtype=np.float64)

    # Extract rotation matrix elements for vectorized operations
    R = rotation_matrices

    # Compute trace for each matrix
    trace = np.trace(R, axis1=1, axis2=2)

    # Case 1: trace > 0 (most common case)
    mask1 = trace > 0
    s1 = np.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
    quaternions[mask1, QuaternionIndex.QW] = 0.25 * s1
    quaternions[mask1, QuaternionIndex.QX] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    quaternions[mask1, QuaternionIndex.QY] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    quaternions[mask1, QuaternionIndex.QZ] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = np.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4 * qx
    quaternions[mask2, QuaternionIndex.QW] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    quaternions[mask2, QuaternionIndex.QX] = 0.25 * s2  # x
    quaternions[mask2, QuaternionIndex.QY] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    quaternions[mask2, QuaternionIndex.QZ] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = np.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4 * qy
    quaternions[mask3, QuaternionIndex.QW] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    quaternions[mask3, QuaternionIndex.QX] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    quaternions[mask3, QuaternionIndex.QY] = 0.25 * s3  # y
    quaternions[mask3, QuaternionIndex.QZ] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = np.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4 * qz
    quaternions[mask4, QuaternionIndex.QW] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    quaternions[mask4, QuaternionIndex.QX] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    quaternions[mask4, QuaternionIndex.QY] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    quaternions[mask4, QuaternionIndex.QZ] = 0.25 * s4  # z

    assert np.all(mask1 | mask2 | mask3 | mask4), "All matrices should fall into one of the four cases."

    return normalize_quaternion_array(quaternions)


def get_quaternion_array_from_rotation_matrix(rotation_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert rotation_matrix.ndim == 2 and rotation_matrix.shape == (3, 3)
    return get_quaternion_array_from_rotation_matrices(rotation_matrix[None, ...])[0]


def get_rotation_matrix_from_euler_array(euler_angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert euler_angles.ndim == 1 and euler_angles.shape[0] == len(EulerAnglesIndex)
    return get_rotation_matrices_from_euler_array(euler_angles[None, ...])[0]


def get_rotation_matrices_from_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert quaternion_array.ndim >= 1 and quaternion_array.shape[-1] == len(QuaternionIndex)

    # Store original shape for reshaping later
    original_shape = quaternion_array.shape[:-1]

    # Flatten to 2D if needed
    if quaternion_array.ndim > 2:
        quaternion_array_ = quaternion_array.reshape(-1, len(QuaternionIndex))
    else:
        quaternion_array_ = quaternion_array

    norm_quaternion = normalize_quaternion_array(quaternion_array_)
    Q_matrices = get_q_matrices(norm_quaternion)
    Q_bar_matrices = get_q_bar_matrices(norm_quaternion)
    rotation_matrix = batch_matmul(Q_matrices, Q_bar_matrices.conj().swapaxes(-1, -2))
    rotation_matrix = rotation_matrix[:, 1:][:, :, 1:]

    # Reshape back to original batch dimensions + (3, 3)
    if len(original_shape) > 1:
        rotation_matrix = rotation_matrix.reshape(original_shape + (3, 3))

    return rotation_matrix


def get_rotation_matrix_from_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert quaternion_array.ndim == 1 and quaternion_array.shape[0] == len(QuaternionIndex)
    return get_rotation_matrices_from_quaternion_array(quaternion_array[None, :])[0]


def get_euler_array_from_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert quaternion_array.ndim >= 1 and quaternion_array.shape[-1] == len(QuaternionIndex)
    norm_quaternion = normalize_quaternion_array(quaternion_array)
    QW, QX, QY, QZ = (
        norm_quaternion[..., QuaternionIndex.QW],
        norm_quaternion[..., QuaternionIndex.QX],
        norm_quaternion[..., QuaternionIndex.QY],
        norm_quaternion[..., QuaternionIndex.QZ],
    )

    euler_angles = np.zeros_like(quaternion_array[..., :3])
    euler_angles[..., EulerAnglesIndex.YAW] = np.arctan2(
        2 * (QW * QZ - QX * QY),
        1 - 2 * (QY**2 + QZ**2),
    )
    euler_angles[..., EulerAnglesIndex.PITCH] = np.arcsin(
        np.clip(2 * (QW * QY + QZ * QX), -1.0, 1.0),
    )
    euler_angles[..., EulerAnglesIndex.ROLL] = np.arctan2(
        2 * (QW * QX - QY * QZ),
        1 - 2 * (QX**2 + QY**2),
    )

    return euler_angles


def conjugate_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the conjugate of an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of conjugated quaternions.
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)
    conjugated_quaternions = np.zeros_like(quaternion_array)
    conjugated_quaternions[..., QuaternionIndex.SCALAR] = quaternion_array[..., QuaternionIndex.SCALAR]
    conjugated_quaternions[..., QuaternionIndex.VECTOR] = -quaternion_array[..., QuaternionIndex.VECTOR]
    return conjugated_quaternions


def invert_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the inverse of an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of inverted quaternions.
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)
    norm_squared = np.sum(quaternion_array**2, axis=-1, keepdims=True)
    assert np.all(norm_squared > 0), "Cannot invert a quaternion with zero norm."
    conjugated_quaternions = conjugate_quaternion_array(quaternion_array)
    inverted_quaternions = conjugated_quaternions / norm_squared
    return inverted_quaternions


def normalize_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalizes an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of normalized quaternions.
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)
    norm = np.linalg.norm(quaternion_array, axis=-1, keepdims=True)
    assert np.all(norm > 0), "Cannot normalize a quaternion with zero norm."
    normalized_quaternions = quaternion_array / norm
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


def get_q_matrices(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the Q matrices for an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of Q matrices.
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)

    qw = quaternion_array[..., QuaternionIndex.QW]
    qx = quaternion_array[..., QuaternionIndex.QX]
    qy = quaternion_array[..., QuaternionIndex.QY]
    qz = quaternion_array[..., QuaternionIndex.QZ]

    batch_shape = quaternion_array.shape[:-1]
    Q_matrices = np.zeros(batch_shape + (4, 4), dtype=np.float64)

    Q_matrices[..., 0, 0] = qw
    Q_matrices[..., 0, 1] = -qx
    Q_matrices[..., 0, 2] = -qy
    Q_matrices[..., 0, 3] = -qz

    Q_matrices[..., 1, 0] = qx
    Q_matrices[..., 1, 1] = qw
    Q_matrices[..., 1, 2] = -qz
    Q_matrices[..., 1, 3] = qy

    Q_matrices[..., 2, 0] = qy
    Q_matrices[..., 2, 1] = qz
    Q_matrices[..., 2, 2] = qw
    Q_matrices[..., 2, 3] = -qx

    Q_matrices[..., 3, 0] = qz
    Q_matrices[..., 3, 1] = -qy
    Q_matrices[..., 3, 2] = qx
    Q_matrices[..., 3, 3] = qw

    return Q_matrices


def get_q_bar_matrices(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the Q-bar matrices for an array of quaternions.
    in the order [qw, qx, qy, qz].
    :param quaternion: Array of quaternions.
    :return: Array of Q-bar matrices.
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)

    qw = quaternion_array[..., QuaternionIndex.QW]
    qx = quaternion_array[..., QuaternionIndex.QX]
    qy = quaternion_array[..., QuaternionIndex.QY]
    qz = quaternion_array[..., QuaternionIndex.QZ]

    batch_shape = quaternion_array.shape[:-1]
    Q_bar_matrices = np.zeros(batch_shape + (4, 4), dtype=np.float64)

    Q_bar_matrices[..., 0, 0] = qw
    Q_bar_matrices[..., 0, 1] = -qx
    Q_bar_matrices[..., 0, 2] = -qy
    Q_bar_matrices[..., 0, 3] = -qz

    Q_bar_matrices[..., 1, 0] = qx
    Q_bar_matrices[..., 1, 1] = qw
    Q_bar_matrices[..., 1, 2] = qz
    Q_bar_matrices[..., 1, 3] = -qy

    Q_bar_matrices[..., 2, 0] = qy
    Q_bar_matrices[..., 2, 1] = -qz
    Q_bar_matrices[..., 2, 2] = qw
    Q_bar_matrices[..., 2, 3] = qx

    Q_bar_matrices[..., 3, 0] = qz
    Q_bar_matrices[..., 3, 1] = qy
    Q_bar_matrices[..., 3, 2] = -qx
    Q_bar_matrices[..., 3, 3] = qw

    return Q_bar_matrices
