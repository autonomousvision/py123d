from typing import Union

import numpy as np
import numpy.typing as npt

from d123.geometry.geometry_index import Vector2DIndex
from d123.geometry.se import StateSE2, StateSE2Index
from d123.geometry.utils.rotation_utils import normalize_angle
from d123.geometry.vector import Vector2D

# TODO: Refactor 2D and 3D transform functions in a more consistent and general way.


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

    rotate_rad = -origin_array[StateSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R_inv = np.array([[cos, -sin], [sin, cos]])

    state_se2_rel = state_se2_array - origin_array
    state_se2_rel[..., StateSE2Index.XY] = state_se2_rel[..., StateSE2Index.XY] @ R_inv.T
    state_se2_rel[..., StateSE2Index.YAW] = normalize_angle(state_se2_rel[..., StateSE2Index.YAW])

    return state_se2_rel


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
    R = np.array([[cos, -sin], [sin, cos]])

    point_2d_rel = point_2d_array - origin_array[..., StateSE2Index.XY]
    point_2d_rel = point_2d_rel @ R.T

    return point_2d_rel


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

    rotate_rad = origin_array[StateSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]])

    state_se2_rel = state_se2_array + origin_array
    state_se2_rel[..., StateSE2Index.XY] = state_se2_rel[..., StateSE2Index.XY] @ R.T
    state_se2_rel[..., StateSE2Index.YAW] = normalize_angle(state_se2_rel[..., StateSE2Index.YAW])

    return state_se2_rel


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
    R = np.array([[cos, -sin], [sin, cos]])

    point_2d_abs = point_2d_array @ R.T
    point_2d_abs = point_2d_abs + origin_array[..., StateSE2Index.XY]

    return point_2d_abs


def translate_se2(state_se2: StateSE2, translation: Vector2D) -> StateSE2:
    """Translate a single SE2 state by a 2D vector.

    :param state_se2: SE2 state to translate
    :param translation: 2D translation vector
    :return: translated SE2 state
    """
    translated_xy = state_se2.array[StateSE2Index.XY] + translation.array[Vector2DIndex.XY]
    return StateSE2(translated_xy[0], translated_xy[1], state_se2.array[StateSE2Index.YAW])


def translate_se2_array(state_se2_array: npt.NDArray[np.float64], translation: Vector2D) -> npt.NDArray[np.float64]:
    """Translate an array of SE2 states by a 2D vector.

    :param state_se2_array: array of SE2 states, indexed by \
        :class:`~d123.geometry.geometry_index.StateSE2Index`, in last dim
    :param translation: 2D translation vector
    :return: translated SE2 array
    """
    result = state_se2_array.copy()
    result[..., StateSE2Index.XY] += translation.array[Vector2DIndex.XY]
    return result


def translate_se2_along_yaw(state_se2: StateSE2, translation: Vector2D) -> StateSE2:
    """Translate a single SE2 state along its local coordinate frame.

    :param state_se2: SE2 state to translate
    :param translation: 2D translation in local frame (x: forward, y: left)
    :return: translated SE2 state
    """
    yaw = state_se2.array[StateSE2Index.YAW]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    # Transform translation from local to global frame
    global_translation = np.array(
        [translation.x * cos_yaw - translation.y * sin_yaw, translation.x * sin_yaw + translation.y * cos_yaw]
    )

    return translate_se2(state_se2, Vector2D.from_array(global_translation))


def translate_se2_array_along_yaw(
    state_se2_array: npt.NDArray[np.float64], translation: Vector2D
) -> npt.NDArray[np.float64]:
    """Translate an array of SE2 states along their respective local coordinate frames.

    :param state_se2_array: array of SE2 states with (x,y,yaw) in last dim
    :param translation: 2D translation in local frame (x: forward, y: left)
    :return: translated SE2 array
    """
    result = state_se2_array.copy()
    yaws = state_se2_array[..., StateSE2Index.YAW]
    cos_yaws, sin_yaws = np.cos(yaws), np.sin(yaws)

    # Transform translation from local to global frame for each state
    global_translation_x = translation.x * cos_yaws - translation.y * sin_yaws
    global_translation_y = translation.x * sin_yaws + translation.y * cos_yaws

    result[..., StateSE2Index.X] += global_translation_x
    result[..., StateSE2Index.Y] += global_translation_y

    return result
