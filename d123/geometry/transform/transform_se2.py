from typing import Union

import numpy as np
import numpy.typing as npt

from d123.geometry.geometry_index import Vector2DIndex
from d123.geometry.se import StateSE2, StateSE2Index
from d123.geometry.utils.rotation_utils import normalize_angle
from d123.geometry.vector import Vector2D


def convert_absolute_to_relative_se2_array(
    origin: Union[StateSE2, npt.NDArray[np.float64]], state_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an StateSE2 array from global to relative coordinates.

    :param origin: origin pose of relative coords system
    :param state_se2_array: array of SE2 states with (x,y,yaw), indexed by \
        :class:`~d123.geometry.geometry_index.StateSE2Index`, in last dim
    :return: SE2 array, index by \
        :class:`~d123.geometry.geometry_index.StateSE2Index`, in last dim
    """
    if isinstance(origin, StateSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    assert len(StateSE2Index) == state_se2_array.shape[-1]

    rotate_rad = -origin_array[StateSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R_inv = np.array([[cos, -sin], [sin, cos]])

    state_se2_rel = state_se2_array - origin_array
    state_se2_rel[..., StateSE2Index.XY] = state_se2_rel[..., StateSE2Index.XY] @ R_inv.T
    state_se2_rel[..., StateSE2Index.YAW] = normalize_angle(state_se2_rel[..., StateSE2Index.YAW])

    return state_se2_rel


def convert_relative_to_absolute_se2_array(
    origin: Union[StateSE2, npt.NDArray[np.float64]], state_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Converts an StateSE2 array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param state_se2_array: array of SE2 states with (x,y,θ) in last dim
    :return: SE2 coords array in relative coordinates
    """
    if isinstance(origin, StateSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    assert len(StateSE2Index) == state_se2_array.shape[-1]

    rotate_rad = origin_array[StateSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]])

    state_se2_abs = np.zeros_like(state_se2_array, dtype=np.float64)
    state_se2_abs[..., StateSE2Index.XY] = state_se2_array[..., StateSE2Index.XY] @ R.T
    state_se2_abs[..., StateSE2Index.XY] += origin_array[..., StateSE2Index.XY]
    state_se2_abs[..., StateSE2Index.YAW] = normalize_angle(
        state_se2_array[..., StateSE2Index.YAW] + origin_array[..., StateSE2Index.YAW]
    )

    return state_se2_abs


def convert_absolute_to_relative_point_2d_array(
    origin: Union[StateSE2, npt.NDArray[np.float64]], point_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts an absolute 2D point array from global to relative coordinates.

    :param origin: origin pose of relative coords system
    :param point_2d_array: array of 2D points with (x,y) in last dim
    :return: 2D points array in relative coordinates
    """
    if isinstance(origin, StateSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    rotate_rad = -origin_array[StateSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    point_2d_rel = point_2d_array - origin_array[..., StateSE2Index.XY]
    point_2d_rel = point_2d_rel @ R.T

    return point_2d_rel


def convert_relative_to_absolute_point_2d_array(
    origin: Union[StateSE2, npt.NDArray[np.float64]], point_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    if isinstance(origin, StateSE2):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE2Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE2 or np.ndarray, got {type(origin)}")

    rotate_rad = origin_array[StateSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    point_2d_abs = point_2d_array @ R.T
    point_2d_abs = point_2d_abs + origin_array[..., StateSE2Index.XY]

    return point_2d_abs


def translate_se2_array_along_body_frame(
    state_se2_array: npt.NDArray[np.float64], translation: Vector2D
) -> npt.NDArray[np.float64]:
    """Translate an array of SE2 states along their respective local coordinate frames.

    :param state_se2_array: array of SE2 states with (x,y,yaw) in last dim
    :param translation: 2D translation in local frame (x: forward, y: left)
    :return: translated SE2 array
    """
    assert len(StateSE2Index) == state_se2_array.shape[-1]
    result = state_se2_array.copy()
    yaws = state_se2_array[..., StateSE2Index.YAW]
    cos_yaws, sin_yaws = np.cos(yaws), np.sin(yaws)

    # Transform translation from local to global frame for each state
    # Create rotation matrices for each state
    R = np.stack([cos_yaws, -sin_yaws, sin_yaws, cos_yaws], axis=-1).reshape(*cos_yaws.shape, 2, 2)

    # Transform translation vector from local to global frame
    translation_vector = translation.array[Vector2DIndex.XY]  # [x, y]
    global_translation = np.einsum("...ij,...j->...i", R, translation_vector)

    result[..., StateSE2Index.XY] += global_translation

    return result


def translate_se2_along_body_frame(state_se2: StateSE2, translation: Vector2D) -> StateSE2:
    """Translate a single SE2 state along its local coordinate frame.

    :param state_se2: SE2 state to translate
    :param translation: 2D translation in local frame (x: forward, y: left)
    :return: translated SE2 state
    """
    return StateSE2.from_array(translate_se2_array_along_body_frame(state_se2.array, translation), copy=False)


def translate_se2_along_x(state_se2: StateSE2, distance: float) -> StateSE2:
    """Translate a single SE2 state along its local X-axis.

    :param state_se2: SE2 state to translate
    :param distance: distance to translate along the local X-axis
    :return: translated SE2 state
    """
    translation = Vector2D.from_array(np.array([distance, 0.0], dtype=np.float64))
    return StateSE2.from_array(translate_se2_array_along_body_frame(state_se2.array, translation), copy=False)


def translate_se2_along_y(state_se2: StateSE2, distance: float) -> StateSE2:
    """Translate a single SE2 state along its local Y-axis.

    :param state_se2: SE2 state to translate
    :param distance: distance to translate along the local Y-axis
    :return: translated SE2 state
    """
    translation = Vector2D.from_array(np.array([0.0, distance], dtype=np.float64))
    return StateSE2.from_array(translate_se2_array_along_body_frame(state_se2.array, translation), copy=False)
