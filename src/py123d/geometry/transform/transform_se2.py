"""SE(2) transformation functions for 2D poses and points.

This module provides functions for converting SE(2) poses and 2D points between
coordinate frames. SE(2) represents rigid-body transformations in 2D, combining
rotation and translation: ``(x, y, yaw)``.

Each operation comes in two variants:

- **Array functions** (e.g. :func:`abs_to_rel_se2_array`) operate on NumPy arrays
  and support batch dimensions.
- **Typed functions** (e.g. :func:`abs_to_rel_se2`) accept and return typed
  :class:`~py123d.geometry.PoseSE2` / :class:`~py123d.geometry.Point2D` objects.

For the 3D counterpart, see :mod:`py123d.geometry.transform.transform_se3`.
"""

import warnings
from typing import Union

import numpy as np
import numpy.typing as npt

from py123d.geometry import Point2D, Point2DIndex, PoseSE2, PoseSE2Index, Vector2D, Vector2DIndex
from py123d.geometry.utils.rotation_utils import normalize_angle


def _extract_pose_se2_array(pose: Union[PoseSE2, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Helper function to extract SE2 pose array from a PoseSE2 or np.ndarray.

    :param pose: Input pose, either a :class:`~py123d.geometry.PoseSE2 instance or a 1D numpy array.
    :raises TypeError: If the input is neither a :class:`~py123d.geometry.PoseSE2 nor a 1D numpy array.
    :return: A 1D numpy array representing the SE2 pose.
    """
    if isinstance(pose, PoseSE2):
        pose_array = pose.array
    elif isinstance(pose, np.ndarray):
        assert pose.ndim == 1 and pose.shape[-1] == len(PoseSE2Index)
        pose_array = pose
    else:
        raise TypeError(f"Expected :class:`~py123d.geometry.PoseSE2 or np.ndarray, got {type(pose)}")
    return pose_array


# ──────────────────────────────────────────────────────────────────────────────
# Array functions: SE2 poses
# ──────────────────────────────────────────────────────────────────────────────


def abs_to_rel_se2_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], pose_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert an SE2 array from absolute to relative coordinates.

    Computes :math:`T_\text{rel} = T_\text{origin}^{-1} \cdot T_\text{abs}` for
    each pose in the array.

    Example::

        >>> origin = PoseSE2(1.0, 2.0, np.pi / 2)
        >>> poses = np.array([[3.0, 4.0, 0.0]], dtype=np.float64)
        >>> rel = abs_to_rel_se2_array(origin, poses)

    :param origin: Origin pose of the relative coordinate system.
    :param pose_se2_array: Array of SE2 poses with shape ``(..., 3)``, indexed by
        :class:`~py123d.geometry.PoseSE2Index` in the last dimension.
    :return: SE2 array in relative coordinates with the same shape as *pose_se2_array*.
    """
    assert len(PoseSE2Index) == pose_se2_array.shape[-1]
    origin_array = _extract_pose_se2_array(origin)

    rotate_rad = -origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R_inv = np.array([[cos, -sin], [sin, cos]])

    pose_se2_rel = pose_se2_array - origin_array
    pose_se2_rel[..., PoseSE2Index.XY] @= R_inv.T
    pose_se2_rel[..., PoseSE2Index.YAW] = normalize_angle(pose_se2_rel[..., PoseSE2Index.YAW])

    return pose_se2_rel


def abs_to_rel_se2(origin: PoseSE2, pose_se2: PoseSE2) -> PoseSE2:
    r"""Convert a single SE2 pose from absolute to relative coordinates.

    Typed wrapper around :func:`abs_to_rel_se2_array`.

    Computes :math:`T_\text{rel} = T_\text{origin}^{-1} \cdot T_\text{abs}`.

    Example::

        >>> origin = PoseSE2(1.0, 1.0, 0.0)
        >>> pose = PoseSE2(2.0, 3.0, 0.0)
        >>> rel = abs_to_rel_se2(origin, pose)  # PoseSE2(1.0, 2.0, 0.0)

    :param origin: Origin pose of the relative coordinate system.
    :param pose_se2: The absolute SE2 pose to convert.
    :return: The SE2 pose in relative coordinates.
    """
    return PoseSE2.from_array(abs_to_rel_se2_array(origin, pose_se2.array), copy=False)


def rel_to_abs_se2_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], pose_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert an SE2 array from relative to absolute coordinates.

    Computes :math:`T_\text{abs} = T_\text{origin} \cdot T_\text{rel}` for each
    pose in the array.

    Example::

        >>> origin = PoseSE2(1.0, 1.0, 0.0)
        >>> rel_poses = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
        >>> abs_poses = rel_to_abs_se2_array(origin, rel_poses)  # [[2.0, 2.0, 0.0]]

    :param origin: Origin pose of the relative coordinate system.
    :param pose_se2_array: Array of SE2 poses with shape ``(..., 3)``, indexed by
        :class:`~py123d.geometry.PoseSE2Index` in the last dimension.
    :return: SE2 array in absolute coordinates with the same shape as *pose_se2_array*.
    """
    assert len(PoseSE2Index) == pose_se2_array.shape[-1]
    origin_array = _extract_pose_se2_array(origin)

    rotate_rad = origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]])

    pose_se2_abs = np.zeros_like(pose_se2_array, dtype=np.float64)
    pose_se2_abs[..., PoseSE2Index.XY] = pose_se2_array[..., PoseSE2Index.XY] @ R.T
    pose_se2_abs[..., PoseSE2Index.XY] += origin_array[..., PoseSE2Index.XY]
    pose_se2_abs[..., PoseSE2Index.YAW] = normalize_angle(
        pose_se2_array[..., PoseSE2Index.YAW] + origin_array[..., PoseSE2Index.YAW]
    )

    return pose_se2_abs


def rel_to_abs_se2(origin: PoseSE2, pose_se2: PoseSE2) -> PoseSE2:
    r"""Convert a single SE2 pose from relative to absolute coordinates.

    Typed wrapper around :func:`rel_to_abs_se2_array`.

    Computes :math:`T_\text{abs} = T_\text{origin} \cdot T_\text{rel}`.

    Example::

        >>> origin = PoseSE2(1.0, 1.0, 0.0)
        >>> rel = PoseSE2(1.0, 1.0, 0.0)
        >>> abs_pose = rel_to_abs_se2(origin, rel)  # PoseSE2(2.0, 2.0, 0.0)

    :param origin: Origin pose of the relative coordinate system.
    :param pose_se2: The relative SE2 pose to convert.
    :return: The SE2 pose in absolute coordinates.
    """
    return PoseSE2.from_array(rel_to_abs_se2_array(origin, pose_se2.array), copy=False)


def reframe_se2_array(
    from_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    pose_se2_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert an SE2 array from one reference frame to another.

    Equivalent to converting from the source frame to absolute coordinates, then
    from absolute coordinates to the target frame:
    ``abs_to_rel(to_origin, rel_to_abs(from_origin, poses))``.

    Example::

        >>> frame_a = PoseSE2(1.0, 0.0, 0.0)
        >>> frame_b = PoseSE2(0.0, 1.0, np.pi / 2)
        >>> poses_in_a = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        >>> poses_in_b = reframe_se2_array(frame_a, frame_b, poses_in_a)

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param pose_se2_array: Array of SE2 poses in the source frame with shape ``(..., 3)``.
    :raises TypeError: If the origins are not :class:`~py123d.geometry.PoseSE2` or ``np.ndarray``.
    :return: The SE2 array in the target frame, indexed by :class:`~py123d.geometry.PoseSE2Index`.
    """
    # Parse from_origin & to_origin
    from_origin_array = _extract_pose_se2_array(from_origin)
    to_origin_array = _extract_pose_se2_array(to_origin)

    assert pose_se2_array.ndim >= 1
    assert pose_se2_array.shape[-1] == len(PoseSE2Index)

    # TODO: Re-write without transforming to absolute frame intermediate step
    abs_array = rel_to_abs_se2_array(from_origin_array, pose_se2_array)
    result_se2_array = abs_to_rel_se2_array(to_origin_array, abs_array)

    return result_se2_array


def reframe_se2(from_origin: PoseSE2, to_origin: PoseSE2, pose_se2: PoseSE2) -> PoseSE2:
    """Convert a single SE2 pose from one reference frame to another.

    Typed wrapper around :func:`reframe_se2_array`.

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param pose_se2: The SE2 pose in the source frame.
    :return: The SE2 pose in the target frame.
    """
    return PoseSE2.from_array(reframe_se2_array(from_origin, to_origin, pose_se2.array), copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# Array functions: 2D points
# ──────────────────────────────────────────────────────────────────────────────


def abs_to_rel_points_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], points_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert a 2D point array from absolute to relative coordinates.

    Computes :math:`p_\text{rel} = R_\text{origin}^T \cdot (p_\text{abs} - t_\text{origin})`
    for each point in the array.

    Example::

        >>> origin = PoseSE2(1.0, 1.0, 0.0)
        >>> points = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        >>> rel_points = abs_to_rel_points_2d_array(origin, points)  # [[1.0, 1.0], [-1.0, 0.0]]

    :param origin: Origin pose of the relative coordinate system.
    :param points_2d_array: Array of 2D points with shape ``(..., 2)``, indexed by
        :class:`~py123d.geometry.Point2DIndex` in the last dimension.
    :return: 2D points in relative coordinates with the same shape as *points_2d_array*.
    """
    assert points_2d_array.ndim >= 1
    assert points_2d_array.shape[-1] == len(Point2DIndex)
    origin_array = _extract_pose_se2_array(origin)

    rotate_rad = -origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    point_2d_rel = points_2d_array - origin_array[..., PoseSE2Index.XY]
    point_2d_rel @= R.T

    return point_2d_rel


def abs_to_rel_point_2d(origin: PoseSE2, point_2d: Point2D) -> Point2D:
    r"""Convert a single 2D point from absolute to relative coordinates.

    Typed wrapper around :func:`abs_to_rel_points_2d_array`.

    Computes :math:`p_\text{rel} = R_\text{origin}^T \cdot (p_\text{abs} - t_\text{origin})`.

    :param origin: Origin pose of the relative coordinate system.
    :param point_2d: The absolute 2D point to convert.
    :return: The 2D point in relative coordinates.
    """
    return Point2D.from_array(abs_to_rel_points_2d_array(origin, point_2d.array), copy=False)


def rel_to_abs_points_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], points_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Convert a relative 2D point array to absolute coordinates.

    Computes :math:`p_\text{abs} = R_\text{origin} \cdot p_\text{rel} + t_\text{origin}`
    for each point in the array.

    :param origin: Origin pose of the relative coordinate system.
    :param points_2d_array: Array of 2D points with shape ``(..., 2)``, indexed by
        :class:`~py123d.geometry.Point2DIndex` in the last dimension.
    :return: 2D points in absolute coordinates with the same shape as *points_2d_array*.
    """

    origin_array = _extract_pose_se2_array(origin)

    rotate_rad = origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    points_2d_abs = points_2d_array @ R.T
    points_2d_abs += origin_array[..., PoseSE2Index.XY]

    return points_2d_abs


def rel_to_abs_point_2d(origin: PoseSE2, point_2d: Point2D) -> Point2D:
    r"""Convert a single 2D point from relative to absolute coordinates.

    Typed wrapper around :func:`rel_to_abs_points_2d_array`.

    Computes :math:`p_\text{abs} = R_\text{origin} \cdot p_\text{rel} + t_\text{origin}`.

    :param origin: Origin pose of the relative coordinate system.
    :param point_2d: The relative 2D point to convert.
    :return: The 2D point in absolute coordinates.
    """
    return Point2D.from_array(rel_to_abs_points_2d_array(origin, point_2d.array), copy=False)


def reframe_points_2d_array(
    from_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    points_2d_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert 2D points from one reference frame to another.

    Equivalent to converting from the source frame to absolute coordinates, then
    from absolute coordinates to the target frame.

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param points_2d_array: Array of 2D points in the source frame with shape ``(..., 2)``.
    :raises TypeError: If the origins are not :class:`~py123d.geometry.PoseSE2` or ``np.ndarray``.
    :return: The 2D points in the target frame, indexed by :class:`~py123d.geometry.Point2DIndex`.
    """

    assert points_2d_array.ndim >= 1
    assert points_2d_array.shape[-1] == len(Point2DIndex)

    from_origin_array = _extract_pose_se2_array(from_origin)
    to_origin_array = _extract_pose_se2_array(to_origin)

    abs_points_array = rel_to_abs_points_2d_array(from_origin_array, points_2d_array)
    result_points_array = abs_to_rel_points_2d_array(to_origin_array, abs_points_array)

    return result_points_array


def reframe_point_2d(from_origin: PoseSE2, to_origin: PoseSE2, point_2d: Point2D) -> Point2D:
    """Convert a single 2D point from one reference frame to another.

    Typed wrapper around :func:`reframe_points_2d_array`.

    :param from_origin: The source origin state in the absolute frame.
    :param to_origin: The target origin state in the absolute frame.
    :param point_2d: The 2D point in the source frame.
    :return: The 2D point in the target frame.
    """
    return Point2D.from_array(reframe_points_2d_array(from_origin, to_origin, point_2d.array), copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# Translation functions
# ──────────────────────────────────────────────────────────────────────────────


def translate_se2_array_along_body_frame(
    pose_se2_array: npt.NDArray[np.float64], translation: Vector2D
) -> npt.NDArray[np.float64]:
    """Translate an array of SE2 states along their respective body frames.

    Each pose is translated by the same *translation* vector, expressed in its local
    coordinate frame (x: forward, y: left). The yaw component is unchanged.

    Example::

        >>> poses = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float64)
        >>> translated = translate_se2_array_along_body_frame(poses, Vector2D(1.0, 0.0))
        >>> # First pose moves +1 in x; second pose (rotated 90 deg) moves +1 in y.

    :param pose_se2_array: Array of SE2 states with shape ``(..., 3)``, indexed by
        :class:`~py123d.geometry.PoseSE2Index` in the last dimension.
    :param translation: 2D translation in the body frame (x: forward, y: left).
    :return: Translated SE2 array with the same shape as *pose_se2_array*.
    """
    assert len(PoseSE2Index) == pose_se2_array.shape[-1]
    result = pose_se2_array.copy()
    yaws = pose_se2_array[..., PoseSE2Index.YAW]
    cos_yaws, sin_yaws = np.cos(yaws), np.sin(yaws)

    # Create rotation matrices for each state
    R = np.stack([cos_yaws, -sin_yaws, sin_yaws, cos_yaws], axis=-1).reshape(*cos_yaws.shape, 2, 2)

    # Transform translation vector from local to global frame
    translation_vector = translation.array[Vector2DIndex.XY]  # [x, y]
    global_translation = np.einsum("...ij,...j->...i", R, translation_vector)

    result[..., PoseSE2Index.XY] += global_translation

    return result


def translate_se2_along_body_frame(pose_se2: PoseSE2, translation: Vector2D) -> PoseSE2:
    """Translate a single SE2 state along its body frame.

    Typed wrapper around :func:`translate_se2_array_along_body_frame`.

    :param pose_se2: SE2 state to translate.
    :param translation: 2D translation in the body frame (x: forward, y: left).
    :return: Translated SE2 state.
    """
    return PoseSE2.from_array(translate_se2_array_along_body_frame(pose_se2.array, translation), copy=False)


def translate_se2_along_x(pose_se2: PoseSE2, distance: float) -> PoseSE2:
    """Translate a single SE2 state along its local X-axis (forward direction).

    Shorthand for ``translate_se2_along_body_frame(pose_se2, Vector2D(distance, 0.0))``.

    :param pose_se2: SE2 state to translate.
    :param distance: Distance to translate along the local X-axis.
    :return: Translated SE2 state.
    """
    translation = Vector2D.from_array(np.array([distance, 0.0], dtype=np.float64))
    return PoseSE2.from_array(translate_se2_array_along_body_frame(pose_se2.array, translation), copy=False)


def translate_se2_along_y(pose_se2: PoseSE2, distance: float) -> PoseSE2:
    """Translate a single SE2 state along its local Y-axis (left direction).

    Shorthand for ``translate_se2_along_body_frame(pose_se2, Vector2D(0.0, distance))``.

    :param pose_se2: SE2 state to translate.
    :param distance: Distance to translate along the local Y-axis.
    :return: Translated SE2 state.
    """
    translation = Vector2D.from_array(np.array([0.0, distance], dtype=np.float64))
    return PoseSE2.from_array(translate_se2_array_along_body_frame(pose_se2.array, translation), copy=False)


def translate_2d_along_body_frame(
    points_2d: npt.NDArray[np.float64],
    yaws: npt.NDArray[np.float64],
    x_translate: npt.NDArray[np.float64],
    y_translate: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Translate 2D points along their respective body frames.

    Unlike :func:`translate_se2_array_along_body_frame`, this function accepts separate
    arrays for orientations and per-point translation distances, which is useful when
    each point has a different translation.

    :param points_2d: Array of 2D points, indexed by :class:`~py123d.geometry.Point2DIndex`.
    :param yaws: Array of yaw angles (one per point).
    :param x_translate: Array of x (forward) translations.
    :param y_translate: Array of y (left) translations.
    :return: Array of translated 2D points, indexed by :class:`~py123d.geometry.Point2DIndex`.
    """
    assert points_2d.shape[-1] == len(Point2DIndex)
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.stack(
        [
            (y_translate * np.cos(yaws + half_pi)) + (x_translate * np.cos(yaws)),
            (y_translate * np.sin(yaws + half_pi)) + (x_translate * np.sin(yaws)),
        ],
        axis=-1,
    )
    return points_2d + translation


# ──────────────────────────────────────────────────────────────────────────────
# Deprecated aliases (remove in next major version)
# ──────────────────────────────────────────────────────────────────────────────


def convert_absolute_to_relative_se2_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], pose_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`abs_to_rel_se2_array` instead."""
    warnings.warn(
        "convert_absolute_to_relative_se2_array is deprecated, use abs_to_rel_se2_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return abs_to_rel_se2_array(origin, pose_se2_array)


def convert_relative_to_absolute_se2_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], pose_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`rel_to_abs_se2_array` instead."""
    warnings.warn(
        "convert_relative_to_absolute_se2_array is deprecated, use rel_to_abs_se2_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rel_to_abs_se2_array(origin, pose_se2_array)


def convert_se2_array_between_origins(
    from_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    se2_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`reframe_se2_array` instead."""
    warnings.warn(
        "convert_se2_array_between_origins is deprecated, use reframe_se2_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reframe_se2_array(from_origin, to_origin, pose_se2_array=se2_array)


def convert_absolute_to_relative_points_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], points_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`abs_to_rel_points_2d_array` instead."""
    warnings.warn(
        "convert_absolute_to_relative_points_2d_array is deprecated, use abs_to_rel_points_2d_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return abs_to_rel_points_2d_array(origin, points_2d_array)


def convert_relative_to_absolute_points_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], points_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`rel_to_abs_points_2d_array` instead."""
    warnings.warn(
        "convert_relative_to_absolute_points_2d_array is deprecated, use rel_to_abs_points_2d_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rel_to_abs_points_2d_array(origin, points_2d_array)


def convert_points_2d_array_between_origins(
    from_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    points_2d_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Deprecated: Use :func:`reframe_points_2d_array` instead."""
    warnings.warn(
        "convert_points_2d_array_between_origins is deprecated, use reframe_points_2d_array instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reframe_points_2d_array(from_origin, to_origin, points_2d_array)
