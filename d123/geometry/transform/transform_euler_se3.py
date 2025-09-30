from typing import Union

import numpy as np
import numpy.typing as npt

from d123.geometry import EulerAngles, EulerStateSE3, EulerStateSE3Index, Point3DIndex, Vector3D, Vector3DIndex
from d123.geometry.utils.rotation_utils import (
    get_rotation_matrices_from_euler_array,
    get_rotation_matrix_from_euler_array,
    normalize_angle,
)


def translate_euler_se3_along_z(state_se3: EulerStateSE3, distance: float) -> EulerStateSE3:

    R = state_se3.rotation_matrix
    z_axis = R[:, 2]

    state_se3_array = state_se3.array.copy()
    state_se3_array[EulerStateSE3Index.XYZ] += distance * z_axis[Vector3DIndex.XYZ]
    return EulerStateSE3.from_array(state_se3_array, copy=False)


def translate_euler_se3_along_y(state_se3: EulerStateSE3, distance: float) -> EulerStateSE3:

    R = state_se3.rotation_matrix
    y_axis = R[:, 1]

    state_se3_array = state_se3.array.copy()
    state_se3_array[EulerStateSE3Index.XYZ] += distance * y_axis[Vector3DIndex.XYZ]
    return EulerStateSE3.from_array(state_se3_array, copy=False)


def translate_euler_se3_along_x(state_se3: EulerStateSE3, distance: float) -> EulerStateSE3:

    R = state_se3.rotation_matrix
    x_axis = R[:, 0]

    state_se3_array = state_se3.array.copy()
    state_se3_array[EulerStateSE3Index.XYZ] += distance * x_axis[Vector3DIndex.XYZ]
    return EulerStateSE3.from_array(state_se3_array, copy=False)


def translate_euler_se3_along_body_frame(state_se3: EulerStateSE3, vector_3d: Vector3D) -> EulerStateSE3:

    R = state_se3.rotation_matrix
    world_translation = R @ vector_3d.array

    state_se3_array = state_se3.array.copy()
    state_se3_array[EulerStateSE3Index.XYZ] += world_translation[Vector3DIndex.XYZ]
    return EulerStateSE3.from_array(state_se3_array, copy=False)


def convert_absolute_to_relative_euler_se3_array(
    origin: Union[EulerStateSE3, npt.NDArray[np.float64]], se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    if isinstance(origin, EulerStateSE3):
        origin_array = origin.array
        t_origin = origin.point_3d.array
        R_origin = origin.rotation_matrix
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(EulerStateSE3Index)
        origin_array = origin
        t_origin = origin_array[EulerStateSE3Index.XYZ]
        R_origin = get_rotation_matrix_from_euler_array(origin_array[EulerStateSE3Index.EULER_ANGLES])
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(origin)}")

    assert se3_array.ndim >= 1
    assert se3_array.shape[-1] == len(EulerStateSE3Index)

    # Prepare output array
    rel_se3_array = se3_array.copy()

    # Vectorized relative position calculation
    abs_positions = se3_array[..., EulerStateSE3Index.XYZ]
    rel_positions = (abs_positions - t_origin) @ R_origin
    rel_se3_array[..., EulerStateSE3Index.XYZ] = rel_positions

    # Convert absolute rotation matrices to relative rotation matrices
    abs_rotation_matrices = get_rotation_matrices_from_euler_array(se3_array[..., EulerStateSE3Index.EULER_ANGLES])
    rel_rotation_matrices = np.einsum("ij,...jk->...ik", R_origin.T, abs_rotation_matrices)
    if se3_array.shape[0] != 0:
        rel_euler_angles = np.array([EulerAngles.from_rotation_matrix(R).array for R in rel_rotation_matrices])
        rel_se3_array[..., EulerStateSE3Index.EULER_ANGLES] = normalize_angle(rel_euler_angles)

    return rel_se3_array


def convert_relative_to_absolute_euler_se3_array(
    origin: EulerStateSE3, se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    if isinstance(origin, EulerStateSE3):
        origin_array = origin.array
        t_origin = origin.point_3d.array
        R_origin = origin.rotation_matrix
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(EulerStateSE3Index)
        origin_array = origin
        t_origin = origin_array[EulerStateSE3Index.XYZ]
        R_origin = get_rotation_matrix_from_euler_array(origin_array[EulerStateSE3Index.EULER_ANGLES])
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(origin)}")

    assert se3_array.ndim >= 1
    assert se3_array.shape[-1] == len(EulerStateSE3Index)

    # Prepare output array
    abs_se3_array = se3_array.copy()

    # Vectorized absolute position calculation: rotate and translate
    rel_positions = se3_array[..., EulerStateSE3Index.XYZ]
    abs_positions = (rel_positions @ R_origin.T) + t_origin
    abs_se3_array[..., EulerStateSE3Index.XYZ] = abs_positions

    # Convert relative rotation matrices to absolute rotation matrices
    rel_rotation_matrices = get_rotation_matrices_from_euler_array(se3_array[..., EulerStateSE3Index.EULER_ANGLES])
    abs_rotation_matrices = np.einsum("ij,...jk->...ik", R_origin, rel_rotation_matrices)

    if se3_array.shape[0] != 0:
        abs_euler_angles = np.array([EulerAngles.from_rotation_matrix(R).array for R in abs_rotation_matrices])
        abs_se3_array[..., EulerStateSE3Index.EULER_ANGLES] = normalize_angle(abs_euler_angles)

    return abs_se3_array


def convert_absolute_to_relative_points_3d_array(
    origin: Union[EulerStateSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    if isinstance(origin, EulerStateSE3):
        t_origin = origin.point_3d.array
        R_origin = origin.rotation_matrix
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(EulerStateSE3Index)
        t_origin = origin[EulerStateSE3Index.XYZ]
        R_origin = get_rotation_matrix_from_euler_array(origin[EulerStateSE3Index.EULER_ANGLES])
    else:
        raise TypeError(f"Expected StateSE3 or np.ndarray, got {type(origin)}")

    assert points_3d_array.ndim >= 1
    assert points_3d_array.shape[-1] == len(Point3DIndex)

    # Translate points to origin frame, then rotate to body frame
    relative_points = (points_3d_array - t_origin) @ R_origin
    return relative_points


def convert_relative_to_absolute_points_3d_array(
    origin: Union[EulerStateSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    if isinstance(origin, EulerStateSE3):
        origin_array = origin.array
    elif isinstance(origin, np.ndarray):
        assert origin.ndim == 1 and origin.shape[-1] == len(EulerStateSE3Index)
        origin_array = origin
    else:
        raise TypeError(f"Expected EulerStateSE3 or np.ndarray, got {type(origin)}")

    assert points_3d_array.shape[-1] == len(Point3DIndex)

    R = EulerAngles.from_array(origin_array[EulerStateSE3Index.EULER_ANGLES]).rotation_matrix
    absolute_points = points_3d_array @ R.T + origin.point_3d.array
    return absolute_points
