from typing import Union

import numpy as np
import numpy.typing as npt

from d123.common.geometry.base import StateSE2, StateSE2Index
from d123.common.geometry.line.polylines import normalize_angle

# TODO: Refactor 2D and 3D transform functions in a more consistent and general way.


def convert_absolute_to_relative_se2_array(
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

    rotate_rad = -origin_array[StateSE2Index.YAW]
    cos, sin = np.cos(rotate_rad), np.sin(rotate_rad)
    R = np.array([[cos, -sin], [sin, cos]])

    state_se2_rel = state_se2_array - origin_array
    state_se2_rel[..., :2] = state_se2_rel[..., :2] @ R.T
    state_se2_rel[..., 2] = normalize_angle(state_se2_rel[..., 2])

    return state_se2_rel


def convert_absolute_to_relative_point_2d_array(
    origin: Union[StateSE2, npt.NDArray[np.float64]], point_2d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Converts an absolute 2D point array from global to relative coordinates.
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
    state_se2_rel[..., :2] = state_se2_rel[..., :2] @ R.T
    state_se2_rel[..., 2] = normalize_angle(state_se2_rel[..., 2])

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
