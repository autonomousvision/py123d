from typing import Union

import numpy as np
import numpy.typing as npt

from d123.geometry import StateSE3, StateSE3Index, Vector3D
from d123.geometry.geometry_index import Vector3DIndex
from d123.geometry.rotation import EulerAngles
from d123.geometry.utils.rotation_utils import (
    get_rotation_matrices_from_euler_array,
    get_rotation_matrix_from_euler_array,
    normalize_angle,
)


def translate_se3_along_z(state_se3: StateSE3, distance: float) -> StateSE3:

    R = state_se3.rotation_matrix
    z_axis = R[:, 2]

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.X] += distance * z_axis[0]
    state_se3_array[StateSE3Index.Y] += distance * z_axis[1]
    state_se3_array[StateSE3Index.Z] += distance * z_axis[2]
    return StateSE3.from_array(state_se3_array)


def translate_se3_along_y(state_se3: StateSE3, distance: float) -> StateSE3:

    R = state_se3.rotation_matrix
    y_axis = R[:, 1]

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.X] += distance * y_axis[0]
    state_se3_array[StateSE3Index.Y] += distance * y_axis[1]
    state_se3_array[StateSE3Index.Z] += distance * y_axis[2]
    return StateSE3.from_array(state_se3_array)


def translate_se3_along_x(state_se3: StateSE3, distance: float) -> StateSE3:

    R = state_se3.rotation_matrix
    x_axis = R[:, 0]

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.X] += distance * x_axis[0]
    state_se3_array[StateSE3Index.Y] += distance * x_axis[1]
    state_se3_array[StateSE3Index.Z] += distance * x_axis[2]

    return StateSE3.from_array(state_se3_array)


def translate_body_frame(state_se3: StateSE3, vector_3d: Vector3D) -> StateSE3:
    R = state_se3.rotation_matrix

    # Transform to world frame
    world_translation = R @ vector_3d.array

    state_se3_array = state_se3.array.copy()
    state_se3_array[StateSE3Index.XYZ] += world_translation[Vector3DIndex.XYZ]

    return StateSE3.from_array(state_se3_array)


def convert_relative_to_absolute_points_3d_array(
    origin: Union[StateSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    # TODO: implement function for origin as np.ndarray
    if isinstance(origin, StateSE3):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE3Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(origin)}")

    R = EulerAngles.from_array(origin_array[StateSE3Index.EULER_ANGLES]).rotation_matrix
    absolute_points = points_3d_array @ R.T + origin.point_3d.array
    return absolute_points


def convert_absolute_to_relative_se3_array(
    origin: Union[StateSE3, npt.NDArray[np.float64]], se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    if isinstance(origin, StateSE3):
        origin_array = origin.array
        t_origin = origin.point_3d.array
        R_origin = origin.rotation_matrix
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE3Index)
        origin_array = origin
        t_origin = origin_array[StateSE3Index.XYZ]
        R_origin = get_rotation_matrix_from_euler_array(origin_array[StateSE3Index.EULER_ANGLES])
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(origin)}")

    assert se3_array.shape[-1] == len(StateSE3Index)

    # Extract positions and orientations from se3_array
    abs_positions = se3_array[..., StateSE3Index.XYZ]
    abs_euler_angles = se3_array[..., StateSE3Index.EULER_ANGLES]

    # Vectorized relative position calculation
    rel_positions = (abs_positions - t_origin) @ R_origin

    # Get rotation matrices for all absolute orientations
    R_abs = get_rotation_matrices_from_euler_array(abs_euler_angles)

    # Compute relative rotations: R_rel = R_origin^T @ R_abs
    np.transpose(R_origin) @ R_abs

    # Convert back to Euler angles (this may need a custom function)
    # For now, using simple subtraction as approximation (this is incorrect for general rotations)
    origin_euler = origin_array[StateSE3Index.EULER_ANGLES]
    rel_euler_angles = abs_euler_angles - origin_euler

    # Prepare output array
    rel_se3_array = se3_array.copy()
    rel_se3_array[..., StateSE3Index.XYZ] = rel_positions
    rel_se3_array[..., StateSE3Index.EULER_ANGLES] = normalize_angle(rel_euler_angles)

    return rel_se3_array


def convert_relative_to_absolute_se3_array(
    origin: StateSE3, se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    if isinstance(origin, StateSE3):
        origin_array = origin.array
        t_origin = origin.point_3d.array
        R_origin = origin.rotation_matrix
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE3Index)
        origin_array = origin
        t_origin = origin_array[StateSE3Index.XYZ]
        R_origin = get_rotation_matrix_from_euler_array(origin_array[StateSE3Index.EULER_ANGLES])
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(origin)}")
    assert se3_array.shape[-1] == len(StateSE3Index)

    # Extract relative positions and orientations
    rel_positions = se3_array[..., StateSE3Index.XYZ]
    rel_euler_angles = se3_array[..., StateSE3Index.EULER_ANGLES]

    # Vectorized absolute position calculation: rotate and translate
    abs_positions = (R_origin @ rel_positions.T).T + t_origin

    # Vectorized absolute orientation: add origin's euler angles
    origin_euler = np.array([origin.roll, origin.pitch, origin.yaw], dtype=np.float64)
    abs_euler_angles = rel_euler_angles + origin_euler

    # Prepare output array
    abs_se3_array = se3_array.copy()
    abs_se3_array[..., StateSE3Index.XYZ] = abs_positions
    abs_se3_array[..., StateSE3Index.EULER_ANGLES] = normalize_angle(abs_euler_angles)

    return abs_se3_array


def convert_absolute_to_relative_points_3d_array(
    origin: Union[StateSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    if isinstance(origin, StateSE3):
        t_origin = origin.point_3d.array
        R_origin = origin.rotation_matrix
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(StateSE3Index)
        t_origin = origin[StateSE3Index.XYZ]
        R_origin = get_rotation_matrix_from_euler_array(origin[StateSE3Index.EULER_ANGLES])
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(origin)}")

    # Translate points to origin frame, then rotate to body frame
    relative_points = (points_3d_array - t_origin) @ R_origin
    return relative_points
