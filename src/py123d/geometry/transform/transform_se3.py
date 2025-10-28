from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.geometry import Point3DIndex, QuaternionIndex, StateSE3, StateSE3Index, Vector3D, Vector3DIndex
from py123d.geometry.utils.rotation_utils import (
    conjugate_quaternion_array,
    get_rotation_matrices_from_quaternion_array,
    get_rotation_matrix_from_quaternion_array,
    multiply_quaternion_arrays,
)


def _extract_rotation_translation_pose_arrays(
    pose: Union[StateSE3, npt.NDArray[np.float64]],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Helper function to extract rotation matrix and translation vector from a StateSE3 or np.ndarray.

    :param pose: A StateSE3 pose or np.ndarray, indexed by :class:`~py123d.geometry.StateSE3Index`.
    :raises TypeError: If the pose is not a StateSE3 or np.ndarray.
    :return: A tuple containing the rotation matrix, translation vector, and pose array.
    """
    if isinstance(pose, StateSE3):
        translation = pose.point_3d.array
        rotation = pose.rotation_matrix
        pose_array = pose.array
    elif isinstance(pose, np.ndarray):
        assert pose.ndim == 1 and pose.shape[-1] == len(StateSE3Index)
        translation = pose[StateSE3Index.XYZ]
        rotation = get_rotation_matrix_from_quaternion_array(pose[StateSE3Index.QUATERNION])
        pose_array = pose
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(pose)}")

    return rotation, translation, pose_array


def convert_absolute_to_relative_points_3d_array(
    origin: Union[StateSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts 3D points from the absolute frame to the relative frame.

    :param origin: The origin state in the absolute frame, as a StateSE3 or np.ndarray.
    :param points_3d_array: The 3D points in the absolute frame.
    :raises TypeError: If the origin is not a StateSE3 or np.ndarray.
    :return: The 3D points in the relative frame, indexed by :class:`~py123d.geometry.Point3DIndex`.
    """

    R_origin, t_origin, _ = _extract_rotation_translation_pose_arrays(origin)

    assert points_3d_array.ndim >= 1
    assert points_3d_array.shape[-1] == len(Point3DIndex)

    # Translate points to origin frame, then rotate to body frame
    relative_points = (points_3d_array - t_origin) @ R_origin
    return relative_points


def convert_absolute_to_relative_se3_array(
    origin: Union[StateSE3, npt.NDArray[np.float64]], se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an SE3 array from the absolute frame to the relative frame.

    :param origin: The origin state in the absolute frame, as a StateSE3 or np.ndarray.
    :param se3_array: The SE3 array in the absolute frame.
    :raises TypeError: If the origin is not a StateSE3 or np.ndarray.
    :return: The SE3 array in the relative frame, indexed by :class:`~py123d.geometry.StateSE3Index`.
    """
    R_origin, t_origin, origin_array = _extract_rotation_translation_pose_arrays(origin)

    assert se3_array.ndim >= 1
    assert se3_array.shape[-1] == len(StateSE3Index)

    abs_positions = se3_array[..., StateSE3Index.XYZ]
    abs_quaternions = se3_array[..., StateSE3Index.QUATERNION]

    rel_se3_array = np.zeros_like(se3_array)

    # 1. Vectorized relative position calculation: translate and rotate
    rel_positions = (abs_positions - t_origin) @ R_origin
    rel_se3_array[..., StateSE3Index.XYZ] = rel_positions

    # 2. Vectorized relative orientation calculation: quaternion multiplication with conjugate
    q_origin_conj = conjugate_quaternion_array(origin_array[StateSE3Index.QUATERNION])
    rel_quaternions = multiply_quaternion_arrays(q_origin_conj, abs_quaternions)

    rel_se3_array[..., StateSE3Index.QUATERNION] = rel_quaternions

    return rel_se3_array


def convert_relative_to_absolute_points_3d_array(
    origin: Union[StateSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts 3D points from the relative frame to the absolute frame.

    :param origin: The origin state in the absolute frame, as a StateSE3 or np.ndarray.
    :param points_3d_array: The 3D points in the relative frame, indexed by :class:`~py123d.geometry.Point3DIndex`.
    :raises TypeError: If the origin is not a StateSE3 or np.ndarray.
    :return: The 3D points in the absolute frame, indexed by :class:`~py123d.geometry.Point3DIndex`.
    """
    R_origin, t_origin, _ = _extract_rotation_translation_pose_arrays(origin)

    assert points_3d_array.shape[-1] == len(Point3DIndex)

    absolute_points = points_3d_array @ R_origin.T + t_origin
    return absolute_points


def convert_relative_to_absolute_se3_array(
    origin: StateSE3, se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an SE3 array from the relative frame to the absolute frame.

    :param origin: The origin state in the relative frame, as a StateSE3 or np.ndarray.
    :param se3_array: The SE3 array in the relative frame.
    :raises TypeError: If the origin is not a StateSE3 or np.ndarray.
    :return: The SE3 array in the absolute frame, indexed by :class:`~py123d.geometry.StateSE3Index`.
    """

    R_origin, t_origin, origin_array = _extract_rotation_translation_pose_arrays(origin)

    assert se3_array.ndim >= 1
    assert se3_array.shape[-1] == len(StateSE3Index)

    # Extract relative positions and orientations
    rel_positions = se3_array[..., StateSE3Index.XYZ]
    rel_quaternions = se3_array[..., StateSE3Index.QUATERNION]

    # Vectorized absolute position calculation: rotate and translate
    abs_positions = (R_origin @ rel_positions.T).T + t_origin
    abs_quaternions = multiply_quaternion_arrays(origin_array[StateSE3Index.QUATERNION], rel_quaternions)

    # Prepare output array
    abs_se3_array = se3_array.copy()
    abs_se3_array[..., StateSE3Index.XYZ] = abs_positions
    abs_se3_array[..., StateSE3Index.QUATERNION] = abs_quaternions

    return abs_se3_array


def convert_se3_array_between_origins(
    from_origin: Union[StateSE3, npt.NDArray[np.float64]],
    to_origin: Union[StateSE3, npt.NDArray[np.float64]],
    se3_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Converts an SE3 array from one origin frame to another origin frame.

    :param from_origin: The source origin state in the absolute frame, as a StateSE3 or np.ndarray.
    :param to_origin: The target origin state in the absolute frame, as a StateSE3 or np.ndarray.
    :param se3_array: The SE3 array in the source origin frame.
    :raises TypeError: If the origins are not StateSE3 or np.ndarray.
    :return: The SE3 array in the target origin frame, indexed by :class:`~py123d.geometry.StateSE3Index`.
    """
    # Parse from_origin & to_origin
    R_from, t_from, from_origin_array = _extract_rotation_translation_pose_arrays(from_origin)
    R_to, t_to, to_origin_array = _extract_rotation_translation_pose_arrays(to_origin)

    assert se3_array.ndim >= 1
    assert se3_array.shape[-1] == len(StateSE3Index)

    rel_positions = se3_array[..., StateSE3Index.XYZ]
    rel_quaternions = se3_array[..., StateSE3Index.QUATERNION]

    # Compute relative transformation: T_to^-1 * T_from
    R_rel = R_to.T @ R_from  # Relative rotation matrix
    t_rel = R_to.T @ (t_from - t_to)  # Relative translation

    q_rel = multiply_quaternion_arrays(
        conjugate_quaternion_array(to_origin_array[StateSE3Index.QUATERNION]),
        from_origin_array[StateSE3Index.QUATERNION],
    )

    # Transform positions: rotate and translate
    new_rel_positions = (R_rel @ rel_positions.T).T + t_rel

    # Transform orientations: quaternion multiplication
    new_rel_quaternions = multiply_quaternion_arrays(q_rel, rel_quaternions)

    # Prepare output array
    result_se3_array = np.zeros_like(se3_array)
    result_se3_array[..., StateSE3Index.XYZ] = new_rel_positions
    result_se3_array[..., StateSE3Index.QUATERNION] = new_rel_quaternions

    return result_se3_array


def convert_points_3d_array_between_origins(
    from_origin: Union[StateSE3, npt.NDArray[np.float64]],
    to_origin: Union[StateSE3, npt.NDArray[np.float64]],
    points_3d_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Converts 3D points from one origin frame to another origin frame.

    :param from_origin: The source origin state in the absolute frame, as a StateSE3 or np.ndarray.
    :param to_origin: The target origin state in the absolute frame, as a StateSE3 or np.ndarray.
    :param points_3d_array: The 3D points in the source origin frame.
    :raises TypeError: If the origins are not StateSE3 or np.ndarray.
    :return: The 3D points in the target origin frame, indexed by :class:`~py123d.geometry.Point3DIndex`.
    """
    # Parse from_origin & to_origin
    R_from, t_from, _ = _extract_rotation_translation_pose_arrays(from_origin)
    R_to, t_to, _ = _extract_rotation_translation_pose_arrays(to_origin)

    assert points_3d_array.ndim >= 1
    assert points_3d_array.shape[-1] == len(Point3DIndex)

    abs_points = points_3d_array @ R_from.T + t_from
    new_rel_points = (abs_points - t_to) @ R_to

    return new_rel_points


def translate_se3_along_z(state_se3: StateSE3, distance: float) -> StateSE3:
    """Translates an SE3 state along the Z-axis.

    :param state_se3: The SE3 state to translate.
    :param distance: The distance to translate along the Z-axis.
    :return: The translated SE3 state.
    """
    R = state_se3.rotation_matrix
    z_axis = R[:, 2]

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.XYZ] += distance * z_axis[Vector3DIndex.XYZ]
    return StateSE3.from_array(state_se3_array, copy=False)


def translate_se3_along_y(state_se3: StateSE3, distance: float) -> StateSE3:
    """Translates a SE3 state along the Y-axis.

    :param state_se3: The SE3 state to translate.
    :param distance: The distance to translate along the Y-axis.
    :return: The translated SE3 state.
    """
    R = state_se3.rotation_matrix
    y_axis = R[:, 1]

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.XYZ] += distance * y_axis[Vector3DIndex.XYZ]
    return StateSE3.from_array(state_se3_array, copy=False)


def translate_se3_along_x(state_se3: StateSE3, distance: float) -> StateSE3:
    """Translates a SE3 state along the X-axis.

    :param state_se3: The SE3 state to translate.
    :param distance: The distance to translate along the X-axis.
    :return: The translated SE3 state.
    """
    R = state_se3.rotation_matrix
    x_axis = R[:, 0]

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.XYZ] += distance * x_axis[Vector3DIndex.XYZ]
    return StateSE3.from_array(state_se3_array, copy=False)


def translate_se3_along_body_frame(state_se3: StateSE3, vector_3d: Vector3D) -> StateSE3:
    """Translates a SE3 state along a vector in the body frame.

    :param state_se3: The SE3 state to translate.
    :param vector_3d: The vector to translate along in the body frame.
    :return: The translated SE3 state.
    """
    R = state_se3.rotation_matrix
    world_translation = R @ vector_3d.array

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.XYZ] += world_translation
    return StateSE3.from_array(state_se3_array, copy=False)


def translate_3d_along_body_frame(
    points_3d: npt.NDArray[np.float64],
    quaternions: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Translates 3D points along a vector in the body frame defined by quaternions.

    :param points_3d: Array of 3D points, index by :class:`~py123d.geometry.Point3DIndex`.
    :param quaternions: Array of quaternions, index by :class:`~py123d.geometry.QuaternionIndex`.
    :param translation: Array of translation vectors, index by :class:`~py123d.geometry.Vector3DIndex`.
    :return: The translated 3D points in the world frame, index by :class:`~py123d.geometry.Point3DIndex`.
    """
    assert points_3d.shape[-1] == len(Point3DIndex)
    assert quaternions.shape[-1] == len(QuaternionIndex)
    assert translation.shape[-1] == len(Vector3DIndex)

    R = get_rotation_matrices_from_quaternion_array(quaternions)
    world_translation = np.einsum("...ij,...j->...i", R, translation)
    return points_3d + world_translation
