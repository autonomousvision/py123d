from typing import Union

import numpy as np
import numpy.typing as npt

from py123d.geometry import Point2DIndex, PoseSE2, PoseSE2Index, Vector2D, Vector2DIndex
from py123d.geometry.utils.rotation_utils import normalize_angle


def _extract_pose_se2_array(pose: Union[PoseSE2, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Helper function to extract SE2 pose array from a PoseSE2 or np.ndarray.

    :param pose: Input pose, either a PoseSE2 instance or a 1D numpy array.
    :raises TypeError: If the input is neither a PoseSE2 nor a 1D numpy array.
    :return: A 1D numpy array representing the SE2 pose.
    """
    if isinstance(pose, PoseSE2):
        pose_array = pose.array
    elif isinstance(pose, np.ndarray):
        assert pose.ndim == 1 and pose.shape[-1] == len(PoseSE2Index)
        pose_array = pose
    else:
        raise TypeError(f"Expected PoseSE2 or np.ndarray, got {type(pose)}")
    return pose_array


def convert_absolute_to_relative_se2_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], pose_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an StateSE2 array from global to relative coordinates.

    :param origin: origin pose of relative coords system
    :param pose_se2_array: array of SE2 poses with (x,y,yaw), indexed by \
        :class:`~py123d.geometry.geometry_index.PoseSE2Index`, in last dim
    :return: SE2 array, index by \
        :class:`~py123d.geometry.geometry_index.PoseSE2Index`, in last dim
    """
    assert len(PoseSE2Index) == pose_se2_array.shape[-1]
    origin_array = _extract_pose_se2_array(origin)

    rotate_rad = -origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R_inv = np.array([[cos, -sin], [sin, cos]])

    pose_se2_rel = pose_se2_array - origin_array
    pose_se2_rel[..., PoseSE2Index.XY] = pose_se2_rel[..., PoseSE2Index.XY] @ R_inv.T
    pose_se2_rel[..., PoseSE2Index.YAW] = normalize_angle(pose_se2_rel[..., PoseSE2Index.YAW])

    return pose_se2_rel


def convert_relative_to_absolute_se2_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], pose_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an StateSE2 array from global to relative coordinates.

    :param origin: origin pose of relative coords system
    :param pose_se2_array: array of SE2 poses with (x,y,θ) in last dim
    :return: SE2 coords array in relative coordinates
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


def convert_se2_array_between_origins(
    from_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    se2_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Converts an SE2 array from one origin frame to another origin frame.

    :param from_origin: The source origin state in the absolute frame, as a PoseSE2 or np.ndarray.
    :param to_origin: The target origin state in the absolute frame, as a PoseSE2 or np.ndarray.
    :param se2_array: The SE2 array in the source origin frame.
    :raises TypeError: If the origins are not PoseSE2 or np.ndarray.
    :return: The SE2 array in the target origin frame, indexed by :class:`~py123d.geometry.PoseSE2Index`.
    """
    # Parse from_origin & to_origin
    from_origin_array = _extract_pose_se2_array(from_origin)
    to_origin_array = _extract_pose_se2_array(to_origin)

    assert se2_array.ndim >= 1
    assert se2_array.shape[-1] == len(PoseSE2Index)

    # TODO: Re-write withouts transforming to absolute frame intermediate step
    abs_array = convert_relative_to_absolute_se2_array(from_origin_array, se2_array)
    result_se2_array = convert_absolute_to_relative_se2_array(to_origin_array, abs_array)

    return result_se2_array


def convert_absolute_to_relative_points_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], points_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an absolute 2D point array from global to relative coordinates.

    :param origin: origin pose of relative coords system
    :param points_2d_array: array of 2D points with (x,y) in last dim
    :return: 2D points array in relative coordinates
    """
    assert points_2d_array.ndim >= 1
    assert points_2d_array.shape[-1] == len(Point2DIndex)
    origin_array = _extract_pose_se2_array(origin)

    rotate_rad = -origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    point_2d_rel = points_2d_array - origin_array[..., PoseSE2Index.XY]
    point_2d_rel = point_2d_rel @ R.T

    return point_2d_rel


def convert_relative_to_absolute_points_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], points_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts relative 2D point array to absolute coordinates.

    :param origin: origin pose of relative coords system
    :param points_2d_array: array of 2D points with (x,y) in last dim
    :return: 2D points array in absolute coordinates
    """

    origin_array = _extract_pose_se2_array(origin)

    rotate_rad = origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    points_2d_abs = points_2d_array @ R.T
    points_2d_abs = points_2d_abs + origin_array[..., PoseSE2Index.XY]

    return points_2d_abs


def convert_points_2d_array_between_origins(
    from_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    to_origin: Union[PoseSE2, npt.NDArray[np.float64]],
    points_2d_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Converts 2D points from one origin frame to another origin frame.

    :param from_origin: The source origin state in the absolute frame, as a PoseSE2 or np.ndarray.
    :param to_origin: The target origin state in the absolute frame, as a PoseSE2 or np.ndarray.
    :param points_2d_array: The 2D points in the source origin frame.
    :raises TypeError: If the origins are not PoseSE2 or np.ndarray.
    :return: The 2D points in the target origin frame, indexed by :class:`~py123d.geometry.Point2DIndex`.
    """

    assert points_2d_array.ndim >= 1
    assert points_2d_array.shape[-1] == len(Point2DIndex)

    from_origin_array = _extract_pose_se2_array(from_origin)
    to_origin_array = _extract_pose_se2_array(to_origin)

    abs_points_array = convert_relative_to_absolute_points_2d_array(from_origin_array, points_2d_array)
    result_points_array = convert_absolute_to_relative_points_2d_array(to_origin_array, abs_points_array)

    return result_points_array


def translate_se2_array_along_body_frame(
    pose_se2_array: npt.NDArray[np.float64], translation: Vector2D
) -> npt.NDArray[np.float64]:
    """Translate an array of SE2 states along their respective local coordinate frames.

    :param pose_se2_array: array of SE2 states with (x,y,yaw) in last dim
    :param translation: 2D translation in local frame (x: forward, y: left)
    :return: translated SE2 array
    """
    assert len(PoseSE2Index) == pose_se2_array.shape[-1]
    result = pose_se2_array.copy()
    yaws = pose_se2_array[..., PoseSE2Index.YAW]
    cos_yaws, sin_yaws = np.cos(yaws), np.sin(yaws)

    # Transform translation from local to global frame for each state
    # Create rotation matrices for each state
    R = np.stack([cos_yaws, -sin_yaws, sin_yaws, cos_yaws], axis=-1).reshape(*cos_yaws.shape, 2, 2)

    # Transform translation vector from local to global frame
    translation_vector = translation.array[Vector2DIndex.XY]  # [x, y]
    global_translation = np.einsum("...ij,...j->...i", R, translation_vector)

    result[..., PoseSE2Index.XY] += global_translation

    return result


def translate_se2_along_body_frame(pose_se2: PoseSE2, translation: Vector2D) -> PoseSE2:
    """Translate a single SE2 state along its local coordinate frame.

    :param pose_se2: SE2 state to translate
    :param translation: 2D translation in local frame (x: forward, y: left)
    :return: translated SE2 state
    """
    return PoseSE2.from_array(translate_se2_array_along_body_frame(pose_se2.array, translation), copy=False)


def translate_se2_along_x(pose_se2: PoseSE2, distance: float) -> PoseSE2:
    """Translate a single SE2 state along its local X-axis.

    :param pose_se2: SE2 state to translate
    :param distance: distance to translate along the local X-axis
    :return: translated SE2 state
    """
    translation = Vector2D.from_array(np.array([distance, 0.0], dtype=np.float64))
    return PoseSE2.from_array(translate_se2_array_along_body_frame(pose_se2.array, translation), copy=False)


def translate_se2_along_y(pose_se2: PoseSE2, distance: float) -> PoseSE2:
    """Translate a single SE2 state along its local Y-axis.

    :param pose_se2: SE2 state to translate
    :param distance: distance to translate along the local Y-axis
    :return: translated SE2 state
    """
    translation = Vector2D.from_array(np.array([0.0, distance], dtype=np.float64))
    return PoseSE2.from_array(translate_se2_array_along_body_frame(pose_se2.array, translation), copy=False)


def translate_2d_along_body_frame(
    points_2d: npt.NDArray[np.float64],
    yaws: npt.NDArray[np.float64],
    x_translate: npt.NDArray[np.float64],
    y_translate: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Translate 2D points along their body frame.

    :param points_2d: Array of 2D points, indexed by :class:`~py123d.geometry.Point2DIndex`.
    :param yaws: Array of yaw angles.
    :param x_translate: Array of x translation, i.e. forward translation.
    :param y_translate: Array of y translation, i.e. left translation.
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
