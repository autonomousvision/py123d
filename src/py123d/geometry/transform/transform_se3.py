"""SE(3) transformation functions for 3D poses and points.

This module provides functions for converting SE(3) poses and 3D points between
coordinate frames. SE(3) represents rigid-body transformations in 3D, combining
rotation (stored as a quaternion) and translation: ``(x, y, z, qw, qx, qy, qz)``.

Each operation comes in two variants:

- **Array functions** (e.g. :func:`abs_to_rel_se3_array`) operate on NumPy arrays
  and support batch dimensions.
- **Typed functions** (e.g. :func:`abs_to_rel_se3`) accept and return typed
  :class:`~py123d.geometry.PoseSE3` / :class:`~py123d.geometry.Point3D` objects.

For the 2D counterpart, see :mod:`py123d.geometry.transform.transform_se2`.
"""

import warnings
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.geometry import Point3D, Point3DIndex, PoseSE3, PoseSE3Index, QuaternionIndex, Vector3D, Vector3DIndex
from py123d.geometry.utils.rotation_utils import (
    conjugate_quaternion_array,
    get_rotation_matrices_from_quaternion_array,
    get_rotation_matrix_from_quaternion_array,
    multiply_quaternion_arrays,
)

# NOTE @DanielDauner: Pre-computed einsum contraction path for the (..., 3) x (3, 3) → (..., 3) pattern.
# Using einsum with this path leverages optimized BLAS dispatch and is significantly
# faster than np.dot / np.matmul for large point clouds (>4 000 points).
_EINSUM_MAT3_PATH = np.einsum_path("...i,ij->...j", np.empty((1, 3)), np.empty((3, 3)), optimize="optimal")[0]
_EINSUM_THRESHOLD = 8000


def _extract_rotation_translation_pose_arrays(
    pose_se3: Union[PoseSE3, npt.NDArray[np.float64]],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Helper function to extract rotation matrix and translation vector from a PoseSE3 or np.ndarray.

    :param pose_se3: A PoseSE3 pose or np.ndarray, indexed by :class:`~py123d.geometry.PoseSE3Index`.
    :raises TypeError: If the pose is not a PoseSE3 or np.ndarray.
    :return: A tuple of ``(rotation_matrix, translation_vector, pose_array)``.
    """
    if isinstance(pose_se3, PoseSE3):
        pose_array = pose_se3.array
        translation = pose_array[PoseSE3Index.XYZ]
        rotation = get_rotation_matrix_from_quaternion_array(pose_array[PoseSE3Index.QUATERNION])
    elif isinstance(pose_se3, np.ndarray):
        assert pose_se3.ndim == 1 and pose_se3.shape[-1] == len(PoseSE3Index)
        pose_array = pose_se3
        translation = pose_se3[PoseSE3Index.XYZ]
        rotation = get_rotation_matrix_from_quaternion_array(pose_se3[PoseSE3Index.QUATERNION])
    else:
        raise TypeError(f"Expected PoseSE3 or np.ndarray, got {type(pose_se3)}")

    return rotation, translation, pose_array


def _matmul_points_3d(points: npt.NDArray[np.float64], matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiply an array of 3D points by a 3x3 matrix: ``points @ matrix``.

    Uses ``np.dot`` for small arrays and ``np.einsum`` with a pre-computed
    contraction path for large arrays (≥ :data:`_EINSUM_THRESHOLD` points).

    :param points: Array of shape ``(..., 3)``.
    :param matrix: Array of shape ``(3, 3)``.
    :return: Transformed points with the same shape as *points*.
    """
    if points.size // 3 < _EINSUM_THRESHOLD:
        result = np.dot(points, matrix)
    elif points.ndim == 2:
        result = np.einsum("ni,ij->nj", points, matrix, optimize=_EINSUM_MAT3_PATH)
    else:
        original_shape = points.shape
        result = np.einsum("ni,ij->nj", points.reshape(-1, 3), matrix, optimize=_EINSUM_MAT3_PATH)
        result = result.reshape(original_shape)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Array functions: 3D points
# ──────────────────────────────────────────────────────────────────────────────


def abs_to_rel_points_3d_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert 3D points from absolute to relative coordinates.

    Computes :math:`p_\text{rel} = R_\text{origin}^T \cdot (p_\text{abs} - t_\text{origin})`
    for each point in the array.

    Example::

        >>> origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([1.0, 2.0, 3.0]))
        >>> points = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
        >>> rel_points = abs_to_rel_points_3d_array(origin, points)  # [[1.0, 1.0, 1.0]]

    :param origin: Origin pose of the relative coordinate system.
    :param points_3d_array: Array of 3D points with shape ``(..., 3)``, indexed by
        :class:`~py123d.geometry.Point3DIndex` in the last dimension.
    :raises TypeError: If *origin* is not a :class:`~py123d.geometry.PoseSE3` or ``np.ndarray``.
    :return: 3D points in relative coordinates with the same shape as *points_3d_array*.
    """

    R_origin, t_origin, _ = _extract_rotation_translation_pose_arrays(origin)

    assert points_3d_array.ndim >= 1
    assert points_3d_array.shape[-1] == len(Point3DIndex)

    if points_3d_array.size // 3 < _EINSUM_THRESHOLD:
        return (points_3d_array - t_origin) @ R_origin

    # Algebraic rewrite: (pts - t) @ R  ==  pts @ R - t @ R
    # Avoids allocating an Nx3 intermediate for the broadcast subtraction.
    t_rel = np.dot(t_origin, R_origin)
    relative_points = _matmul_points_3d(points_3d_array, R_origin)
    relative_points -= t_rel
    return relative_points


def abs_to_rel_point_3d(origin: PoseSE3, point_3d: Point3D) -> Point3D:
    r"""Convert a single 3D point from absolute to relative coordinates.

    Typed wrapper around :func:`abs_to_rel_points_3d_array`.

    Computes :math:`p_\text{rel} = R_\text{origin}^T \cdot (p_\text{abs} - t_\text{origin})`.

    :param origin: The origin pose of the relative coordinate system.
    :param point_3d: The absolute 3D point to convert.
    :return: The 3D point in relative coordinates.
    """
    return Point3D.from_array(abs_to_rel_points_3d_array(origin, point_3d.array), copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# Array functions: SE3 poses
# ──────────────────────────────────────────────────────────────────────────────


def abs_to_rel_se3_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], pose_se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert an SE3 array from absolute to relative coordinates.

    Computes :math:`T_\text{rel} = T_\text{origin}^{-1} \cdot T_\text{abs}` for
    each pose in the array. Positions are transformed via the rotation matrix and
    orientations via quaternion multiplication.

    Example::

        >>> origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, np.pi / 2), np.array([1.0, 0.0, 0.0]))
        >>> poses = np.zeros((1, len(PoseSE3Index)), dtype=np.float64)
        >>> poses[0, PoseSE3Index.QW] = 1.0  # identity rotation
        >>> rel = abs_to_rel_se3_array(origin, poses)

    :param origin: Origin pose of the relative coordinate system.
    :param pose_se3_array: Array of SE3 poses with shape ``(..., 7)``, indexed by
        :class:`~py123d.geometry.PoseSE3Index` in the last dimension.
    :raises TypeError: If *origin* is not a :class:`~py123d.geometry.PoseSE3` or ``np.ndarray``.
    :return: SE3 array in relative coordinates with the same shape as *pose_se3_array*.
    """
    R_origin, t_origin, origin_array = _extract_rotation_translation_pose_arrays(origin)

    assert pose_se3_array.ndim >= 1
    assert pose_se3_array.shape[-1] == len(PoseSE3Index)

    abs_positions = pose_se3_array[..., PoseSE3Index.XYZ]
    abs_quaternions = pose_se3_array[..., PoseSE3Index.QUATERNION]

    q_origin_conj = conjugate_quaternion_array(origin_array[PoseSE3Index.QUATERNION])

    if abs_positions.size // 3 < _EINSUM_THRESHOLD:
        rel_positions = (abs_positions - t_origin) @ R_origin
    else:
        t_rel = np.dot(t_origin, R_origin)
        rel_positions = _matmul_points_3d(abs_positions, R_origin)
        rel_positions -= t_rel

    rel_quaternions = multiply_quaternion_arrays(q_origin_conj, abs_quaternions)

    rel_se3_array = np.empty_like(pose_se3_array)
    rel_se3_array[..., PoseSE3Index.XYZ] = rel_positions
    rel_se3_array[..., PoseSE3Index.QUATERNION] = rel_quaternions
    return rel_se3_array


def abs_to_rel_se3(origin: PoseSE3, pose_se3: PoseSE3) -> PoseSE3:
    r"""Convert a single SE3 pose from absolute to relative coordinates.

    Typed wrapper around :func:`abs_to_rel_se3_array`.

    Computes :math:`T_\text{rel} = T_\text{origin}^{-1} \cdot T_\text{abs}`.

    :param origin: The origin pose of the relative coordinate system.
    :param pose_se3: The absolute SE3 pose to convert.
    :return: The SE3 pose in relative coordinates.
    """
    return PoseSE3.from_array(abs_to_rel_se3_array(origin, pose_se3.array), copy=False)


def rel_to_abs_points_3d_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert 3D points from relative to absolute coordinates.

    Computes :math:`p_\text{abs} = R_\text{origin} \cdot p_\text{rel} + t_\text{origin}`
    for each point in the array.

    :param origin: Origin pose of the relative coordinate system.
    :param points_3d_array: Array of 3D points with shape ``(..., 3)``, indexed by
        :class:`~py123d.geometry.Point3DIndex` in the last dimension.
    :raises TypeError: If *origin* is not a :class:`~py123d.geometry.PoseSE3` or ``np.ndarray``.
    :return: 3D points in absolute coordinates with the same shape as *points_3d_array*.
    """
    R_origin, t_origin, _ = _extract_rotation_translation_pose_arrays(origin)

    assert points_3d_array.shape[-1] == len(Point3DIndex)

    if points_3d_array.size // 3 < _EINSUM_THRESHOLD:
        absolute_points = points_3d_array @ R_origin.T + t_origin
    else:
        absolute_points = _matmul_points_3d(points_3d_array, R_origin.T)
        absolute_points += t_origin
    return absolute_points


def rel_to_abs_point_3d(origin: PoseSE3, point_3d: Point3D) -> Point3D:
    r"""Convert a single 3D point from relative to absolute coordinates.

    Typed wrapper around :func:`rel_to_abs_points_3d_array`.

    Computes :math:`p_\text{abs} = R_\text{origin} \cdot p_\text{rel} + t_\text{origin}`.

    :param origin: The origin pose of the relative coordinate system.
    :param point_3d: The relative 3D point to convert.
    :return: The 3D point in absolute coordinates.
    """
    return Point3D.from_array(rel_to_abs_points_3d_array(origin, point_3d.array), copy=False)


def rel_to_abs_se3_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], pose_se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert an SE3 array from relative to absolute coordinates.

    Computes :math:`T_\text{abs} = T_\text{origin} \cdot T_\text{rel}` for each
    pose in the array.

    :param origin: Origin pose of the relative coordinate system.
    :param pose_se3_array: Array of SE3 poses with shape ``(..., 7)``, indexed by
        :class:`~py123d.geometry.PoseSE3Index` in the last dimension.
    :raises TypeError: If *origin* is not a :class:`~py123d.geometry.PoseSE3` or ``np.ndarray``.
    :return: SE3 array in absolute coordinates with the same shape as *pose_se3_array*.
    """

    R_origin, t_origin, origin_array = _extract_rotation_translation_pose_arrays(origin)

    assert pose_se3_array.ndim >= 1
    assert pose_se3_array.shape[-1] == len(PoseSE3Index)

    rel_positions = pose_se3_array[..., PoseSE3Index.XYZ]
    rel_quaternions = pose_se3_array[..., PoseSE3Index.QUATERNION]

    if rel_positions.size // 3 < _EINSUM_THRESHOLD:
        abs_positions = rel_positions @ R_origin.T + t_origin
    else:
        abs_positions = _matmul_points_3d(rel_positions, R_origin.T)
        abs_positions += t_origin

    abs_quaternions = multiply_quaternion_arrays(origin_array[PoseSE3Index.QUATERNION], rel_quaternions)

    abs_se3_array = np.empty_like(pose_se3_array)
    abs_se3_array[..., PoseSE3Index.XYZ] = abs_positions
    abs_se3_array[..., PoseSE3Index.QUATERNION] = abs_quaternions
    return abs_se3_array


def rel_to_abs_se3(origin: PoseSE3, pose_se3: PoseSE3) -> PoseSE3:
    r"""Convert a single SE3 pose from relative to absolute coordinates.

    Typed wrapper around :func:`rel_to_abs_se3_array`.

    Computes :math:`T_\text{abs} = T_\text{origin} \cdot T_\text{rel}`.

    :param origin: The origin pose of the relative coordinate system.
    :param pose_se3: The relative SE3 pose to convert.
    :return: The SE3 pose in absolute coordinates.
    """
    return PoseSE3.from_array(rel_to_abs_se3_array(origin, pose_se3.array), copy=False)


def reframe_se3_array(
    from_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    pose_se3_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert an SE3 array from one reference frame to another.

    Equivalent to converting from the source frame to absolute coordinates, then
    from absolute coordinates to the target frame:
    ``abs_to_rel(to_origin, rel_to_abs(from_origin, poses))``, but computed more
    efficiently as a single relative transformation.

    Example::

        >>> frame_a = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([1.0, 0.0, 0.0]))
        >>> frame_b = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, np.pi / 2), np.array([0.0, 1.0, 0.0]))
        >>> poses_in_a = np.zeros((1, len(PoseSE3Index)), dtype=np.float64)
        >>> poses_in_a[0, PoseSE3Index.QW] = 1.0
        >>> poses_in_b = reframe_se3_array(frame_a, frame_b, poses_in_a)

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param pose_se3_array: Array of SE3 poses in the source frame with shape ``(..., 7)``.
    :raises TypeError: If the origins are not :class:`~py123d.geometry.PoseSE3` or ``np.ndarray``.
    :return: The SE3 array in the target frame, indexed by :class:`~py123d.geometry.PoseSE3Index`.
    """
    # Parse from_origin & to_origin
    R_from, t_from, from_origin_array = _extract_rotation_translation_pose_arrays(from_origin)
    R_to, t_to, to_origin_array = _extract_rotation_translation_pose_arrays(to_origin)

    assert pose_se3_array.ndim >= 1
    assert pose_se3_array.shape[-1] == len(PoseSE3Index)

    # Compute relative transformation: T_to^-1 * T_from
    R_rel = R_to.T @ R_from  # Relative rotation matrix
    t_rel = R_to.T @ (t_from - t_to)  # Relative translation

    q_rel = multiply_quaternion_arrays(
        conjugate_quaternion_array(to_origin_array[PoseSE3Index.QUATERNION]),
        from_origin_array[PoseSE3Index.QUATERNION],
    )

    rel_positions = pose_se3_array[..., PoseSE3Index.XYZ]
    rel_quaternions = pose_se3_array[..., PoseSE3Index.QUATERNION]

    # Transform positions: rotate and translate
    if rel_positions.size // 3 < _EINSUM_THRESHOLD:
        new_rel_positions = rel_positions @ R_rel.T + t_rel
    else:
        new_rel_positions = _matmul_points_3d(rel_positions, R_rel.T)
        new_rel_positions += t_rel

    # Transform orientations: quaternion multiplication
    new_rel_quaternions = multiply_quaternion_arrays(q_rel, rel_quaternions)

    # Prepare output array
    result_se3_array = np.empty_like(pose_se3_array)
    result_se3_array[..., PoseSE3Index.XYZ] = new_rel_positions
    result_se3_array[..., PoseSE3Index.QUATERNION] = new_rel_quaternions
    return result_se3_array


def reframe_se3(from_origin: PoseSE3, to_origin: PoseSE3, pose_se3: PoseSE3) -> PoseSE3:
    """Convert a single SE3 pose from one reference frame to another.

    Typed wrapper around :func:`reframe_se3_array`.

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param pose_se3: The SE3 pose in the source frame.
    :return: The SE3 pose in the target frame.
    """
    return PoseSE3.from_array(reframe_se3_array(from_origin, to_origin, pose_se3.array), copy=False)


def reframe_points_3d_array(
    from_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    points_3d_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert 3D points from one reference frame to another.

    Equivalent to converting from the source frame to absolute coordinates, then
    from absolute coordinates to the target frame, but computed more efficiently
    as a single relative transformation.

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param points_3d_array: Array of 3D points in the source frame with shape ``(..., 3)``.
    :raises TypeError: If the origins are not :class:`~py123d.geometry.PoseSE3` or ``np.ndarray``.
    :return: The 3D points in the target frame, indexed by :class:`~py123d.geometry.Point3DIndex`.
    """
    # Parse from_origin & to_origin
    R_from, t_from, _ = _extract_rotation_translation_pose_arrays(from_origin)
    R_to, t_to, _ = _extract_rotation_translation_pose_arrays(to_origin)

    assert points_3d_array.ndim >= 1
    assert points_3d_array.shape[-1] == len(Point3DIndex)

    R_rel = R_to.T @ R_from  # Relative rotation matrix
    t_rel = R_to.T @ (t_from - t_to)  # Relative translation

    if points_3d_array.size // 3 < _EINSUM_THRESHOLD:
        conv_points_3d_array = points_3d_array @ R_rel.T + t_rel
    else:
        conv_points_3d_array = _matmul_points_3d(points_3d_array, R_rel.T)
        conv_points_3d_array += t_rel
    return conv_points_3d_array


def reframe_point_3d(from_origin: PoseSE3, to_origin: PoseSE3, point_3d: Point3D) -> Point3D:
    """Convert a single 3D point from one reference frame to another.

    Typed wrapper around :func:`reframe_points_3d_array`.

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param point_3d: The 3D point in the source frame.
    :return: The 3D point in the target frame.
    """
    return Point3D.from_array(reframe_points_3d_array(from_origin, to_origin, point_3d.array), copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# Translation functions
# ──────────────────────────────────────────────────────────────────────────────


def translate_se3_along_z(pose_se3: PoseSE3, distance: float) -> PoseSE3:
    """Translate an SE3 state along its local Z-axis (up direction).

    Shorthand for ``translate_se3_along_body_frame(pose_se3, Vector3D(0.0, 0.0, distance))``.

    :param pose_se3: The SE3 state to translate.
    :param distance: The distance to translate along the local Z-axis.
    :return: The translated SE3 state.
    """
    R = pose_se3.rotation_matrix
    z_axis = R[:, 2]

    pose_se3_array = pose_se3.array.copy()
    pose_se3_array[PoseSE3Index.XYZ] += distance * z_axis[Vector3DIndex.XYZ]
    return PoseSE3.from_array(pose_se3_array, copy=False)


def translate_se3_along_y(pose_se3: PoseSE3, distance: float) -> PoseSE3:
    """Translate an SE3 state along its local Y-axis (left direction).

    Shorthand for ``translate_se3_along_body_frame(pose_se3, Vector3D(0.0, distance, 0.0))``.

    :param pose_se3: The SE3 state to translate.
    :param distance: The distance to translate along the local Y-axis.
    :return: The translated SE3 state.
    """
    R = pose_se3.rotation_matrix
    y_axis = R[:, 1]

    pose_se3_array = pose_se3.array.copy()
    pose_se3_array[PoseSE3Index.XYZ] += distance * y_axis[Vector3DIndex.XYZ]
    return PoseSE3.from_array(pose_se3_array, copy=False)


def translate_se3_along_x(pose_se3: PoseSE3, distance: float) -> PoseSE3:
    """Translate an SE3 state along its local X-axis (forward direction).

    Shorthand for ``translate_se3_along_body_frame(pose_se3, Vector3D(distance, 0.0, 0.0))``.

    :param pose_se3: The SE3 state to translate.
    :param distance: The distance to translate along the local X-axis.
    :return: The translated SE3 state.
    """
    R = pose_se3.rotation_matrix
    x_axis = R[:, 0]

    pose_se3_array = pose_se3.array.copy()
    pose_se3_array[PoseSE3Index.XYZ] += distance * x_axis[Vector3DIndex.XYZ]
    return PoseSE3.from_array(pose_se3_array, copy=False)


def translate_se3_along_body_frame(pose_se3: PoseSE3, translation: Vector3D) -> PoseSE3:
    """Translate an SE3 state along a vector in its body frame.

    The *translation* vector is rotated from the body frame into the world frame
    and added to the position. The orientation is unchanged.

    :param pose_se3: The SE3 state to translate.
    :param translation: The 3D translation vector in the body frame.
    :return: The translated SE3 state.
    """
    R = pose_se3.rotation_matrix
    world_translation = R @ translation.array

    pose_se3_array = pose_se3.array.copy()
    pose_se3_array[PoseSE3Index.XYZ] += world_translation
    return PoseSE3.from_array(pose_se3_array, copy=False)


def translate_3d_along_body_frame(
    points_3d: npt.NDArray[np.float64],
    quaternions: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Translate 3D points along their respective body frames defined by quaternions.

    Unlike :func:`translate_se3_along_body_frame`, this function operates on raw
    arrays and supports per-point translations via broadcasting.

    :param points_3d: Array of 3D points, indexed by :class:`~py123d.geometry.Point3DIndex`.
    :param quaternions: Array of quaternions, indexed by :class:`~py123d.geometry.QuaternionIndex`.
    :param translation: Array of translation vectors, indexed by :class:`~py123d.geometry.Vector3DIndex`.
    :return: Translated 3D points in the world frame, indexed by :class:`~py123d.geometry.Point3DIndex`.
    """
    assert points_3d.shape[-1] == len(Point3DIndex)
    assert quaternions.shape[-1] == len(QuaternionIndex)
    assert translation.shape[-1] == len(Vector3DIndex)

    R = get_rotation_matrices_from_quaternion_array(quaternions)
    world_translation = np.einsum("...ij,...j->...i", R, translation)
    return points_3d + world_translation


# ──────────────────────────────────────────────────────────────────────────────
# Deprecated aliases (remove in next major version)
# ──────────────────────────────────────────────────────────────────────────────


def convert_absolute_to_relative_points_3d_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`abs_to_rel_points_3d_array` instead."""
    warnings.warn(
        "convert_absolute_to_relative_points_3d_array is deprecated, use abs_to_rel_points_3d_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return abs_to_rel_points_3d_array(origin, points_3d_array)


def convert_absolute_to_relative_se3_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`abs_to_rel_se3_array` instead."""
    warnings.warn(
        "convert_absolute_to_relative_se3_array is deprecated, use abs_to_rel_se3_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return abs_to_rel_se3_array(origin, pose_se3_array=se3_array)


def convert_relative_to_absolute_points_3d_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`rel_to_abs_points_3d_array` instead."""
    warnings.warn(
        "convert_relative_to_absolute_points_3d_array is deprecated, use rel_to_abs_points_3d_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rel_to_abs_points_3d_array(origin, points_3d_array)


def convert_relative_to_absolute_se3_array(
    origin: Union[PoseSE3, npt.NDArray[np.float64]], se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`rel_to_abs_se3_array` instead."""
    warnings.warn(
        "convert_relative_to_absolute_se3_array is deprecated, use rel_to_abs_se3_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rel_to_abs_se3_array(origin, pose_se3_array=se3_array)


def convert_se3_array_between_origins(
    from_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    se3_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`reframe_se3_array` instead."""
    warnings.warn(
        "convert_se3_array_between_origins is deprecated, use reframe_se3_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reframe_se3_array(from_origin, to_origin, pose_se3_array=se3_array)


def convert_points_3d_array_between_origins(
    from_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE3, npt.NDArray[np.float64]],
    points_3d_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`reframe_points_3d_array` instead."""
    warnings.warn(
        "convert_points_3d_array_between_origins is deprecated, use reframe_points_3d_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reframe_points_3d_array(from_origin, to_origin, points_3d_array)
