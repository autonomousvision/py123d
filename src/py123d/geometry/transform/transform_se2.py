from typing import Union

import numpy as np
import numpy.typing as npt

from py123d.geometry import Point2DIndex, PoseSE2, PoseSE2Index, Vector2D, Vector2DIndex
from py123d.geometry.utils.rotation_utils import normalize_angle


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
    if isinstance(origin, PoseSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(PoseSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    assert len(PoseSE2Index) == pose_se2_array.shape[-1]

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
    """
    Converts an StateSE2 array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param pose_se2_array: array of SE2 poses with (x,y,θ) in last dim
    :return: SE2 coords array in relative coordinates
    """
    if isinstance(origin, PoseSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(PoseSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    assert len(PoseSE2Index) == pose_se2_array.shape[-1]

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


def convert_absolute_to_relative_point_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], point_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an absolute 2D point array from global to relative coordinates.

    :param origin: origin pose of relative coords system
    :param point_2d_array: array of 2D points with (x,y) in last dim
    :return: 2D points array in relative coordinates
    """
    if isinstance(origin, PoseSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(PoseSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    rotate_rad = -origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    point_2d_rel = point_2d_array - origin_array[..., PoseSE2Index.XY]
    point_2d_rel = point_2d_rel @ R.T

    return point_2d_rel


def convert_relative_to_absolute_point_2d_array(
    origin: Union[PoseSE2, npt.NDArray[np.float64]], point_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    if isinstance(origin, PoseSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(PoseSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    rotate_rad = origin_array[PoseSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    point_2d_abs = point_2d_array @ R.T
    point_2d_abs = point_2d_abs + origin_array[..., PoseSE2Index.XY]

    return point_2d_abs


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
