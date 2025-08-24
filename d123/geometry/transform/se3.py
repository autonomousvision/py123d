import numpy as np
import numpy.typing as npt

from d123.geometry.base import Point3DIndex, StateSE3, StateSE3Index
from d123.geometry.vector import Vector3D

# def get_rotation_matrix(state_se3: StateSE3) -> npt.NDArray[np.float64]:
#     R_x = np.array(
#         [
#             [1, 0, 0],
#             [0, np.cos(state_se3.roll), -np.sin(state_se3.roll)],
#             [0, np.sin(state_se3.roll), np.cos(state_se3.roll)],
#         ],
#         dtype=np.float64,
#     )
#     R_y = np.array(
#         [
#             [np.cos(state_se3.pitch), 0, np.sin(state_se3.pitch)],
#             [0, 1, 0],
#             [-np.sin(state_se3.pitch), 0, np.cos(state_se3.pitch)],
#         ],
#         dtype=np.float64,
#     )
#     R_z = np.array(
#         [
#             [np.cos(state_se3.yaw), -np.sin(state_se3.yaw), 0],
#             [np.sin(state_se3.yaw), np.cos(state_se3.yaw), 0],
#             [0, 0, 1],
#         ],
#         dtype=np.float64,
#     )
#     return R_z @ R_y @ R_x


def get_rotation_matrix(state_se3: StateSE3) -> npt.NDArray[np.float64]:
    # Intrinsic Z-Y'-X'' rotation: R = R_x(roll) @ R_y(pitch) @ R_z(yaw)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(state_se3.roll), -np.sin(state_se3.roll)],
            [0, np.sin(state_se3.roll), np.cos(state_se3.roll)],
        ],
        dtype=np.float64,
    )
    R_y = np.array(
        [
            [np.cos(state_se3.pitch), 0, np.sin(state_se3.pitch)],
            [0, 1, 0],
            [-np.sin(state_se3.pitch), 0, np.cos(state_se3.pitch)],
        ],
        dtype=np.float64,
    )
    R_z = np.array(
        [
            [np.cos(state_se3.yaw), -np.sin(state_se3.yaw), 0],
            [np.sin(state_se3.yaw), np.cos(state_se3.yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return R_x @ R_y @ R_z


def translate_se3_along_z(state_se3: StateSE3, distance: float) -> StateSE3:

    R = get_rotation_matrix(state_se3)
    z_axis = R[:, 2]

    new_x = state_se3.x + distance * z_axis[0]
    new_y = state_se3.y + distance * z_axis[1]
    new_z = state_se3.z + distance * z_axis[2]

    return StateSE3(new_x, new_y, new_z, state_se3.roll, state_se3.pitch, state_se3.yaw)


def translate_se3_along_y(state_se3: StateSE3, distance: float) -> StateSE3:

    R = get_rotation_matrix(state_se3)
    y_axis = R[:, 1]

    new_x = state_se3.x + distance * y_axis[0]
    new_y = state_se3.y + distance * y_axis[1]
    new_z = state_se3.z + distance * y_axis[2]

    return StateSE3(new_x, new_y, new_z, state_se3.roll, state_se3.pitch, state_se3.yaw)


def translate_se3_along_x(state_se3: StateSE3, distance: float) -> StateSE3:

    R = get_rotation_matrix(state_se3)
    x_axis = R[:, 0]

    new_x = state_se3.x + distance * x_axis[0]
    new_y = state_se3.y + distance * x_axis[1]
    new_z = state_se3.z + distance * x_axis[2]

    return StateSE3(new_x, new_y, new_z, state_se3.roll, state_se3.pitch, state_se3.yaw)


def translate_body_frame(state_se3: StateSE3, vector_3d: Vector3D) -> StateSE3:
    R = get_rotation_matrix(state_se3)

    body_translation = vector_3d.array

    # Transform to world frame
    world_translation = R @ body_translation

    return StateSE3(
        state_se3.x + world_translation[0],
        state_se3.y + world_translation[1],
        state_se3.z + world_translation[2],
        state_se3.roll,
        state_se3.pitch,
        state_se3.yaw,
    )


def convert_relative_to_absolute_points_3d_array(
    origin: StateSE3, points_3d_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    # TODO: implement function for origin as np.ndarray

    R = get_rotation_matrix(origin)
    absolute_points = points_3d_array @ R.T + origin.point_3d.array
    return absolute_points


def convert_absolute_to_relative_se3_array(
    origin: StateSE3, se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    assert se3_array.shape[-1] == len(StateSE3Index)
    # TODO: remove transform for-loop, use vectorized operations

    # Extract rotation and translation of origin
    R_origin = get_rotation_matrix(origin)
    t_origin = origin.point_3d.array

    # Prepare output array
    rel_se3_array = np.empty_like(se3_array)

    # For each SE3 in the array
    for i in range(se3_array.shape[0]):
        abs_se3 = se3_array[i]
        abs_pos = abs_se3[StateSE3Index.XYZ]
        abs_rpy = abs_se3[StateSE3Index.ROLL : StateSE3Index.YAW + 1]

        # Relative position: rotate and translate
        rel_pos = R_origin.T @ (abs_pos - t_origin)

        # Relative orientation: subtract origin's rpy
        rel_rpy = abs_rpy - np.array([origin.roll, origin.pitch, origin.yaw], dtype=np.float64)

        rel_se3_array[i, StateSE3Index.X : StateSE3Index.Z + 1] = rel_pos
        rel_se3_array[i, StateSE3Index.ROLL : StateSE3Index.YAW + 1] = rel_rpy

    return rel_se3_array


def convert_relative_to_absolute_se3_array(
    origin: StateSE3, se3_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    assert se3_array.shape[-1] == len(StateSE3Index)
    # TODO: remove transform for-loop, use vectorized operations

    # Extract rotation and translation of origin
    R_origin = get_rotation_matrix(origin)
    t_origin = origin.point_3d.array

    # Prepare output array
    abs_se3_array = np.empty_like(se3_array)

    # For each SE3 in the array
    for i in range(se3_array.shape[0]):
        rel_se3 = se3_array[i]
        rel_pos = rel_se3[StateSE3Index.XYZ]
        rel_rpy = rel_se3[StateSE3Index.ROLL : StateSE3Index.YAW + 1]

        # Absolute position: rotate and translate
        abs_pos = R_origin @ rel_pos + t_origin

        # Absolute orientation: add origin's rpy
        abs_rpy = rel_rpy + np.array([origin.roll, origin.pitch, origin.yaw], dtype=np.float64)

        abs_se3_array[i, StateSE3Index.X : StateSE3Index.Z + 1] = abs_pos
        abs_se3_array[i, StateSE3Index.ROLL : StateSE3Index.YAW + 1] = abs_rpy

    return abs_se3_array


def translate_points_3d_along_z(
    state_se3: StateSE3,
    points_3d: npt.NDArray[np.float64],
    distance: float,
) -> npt.NDArray[np.float64]:
    assert points_3d.shape[-1] == len(Point3DIndex)

    R = get_rotation_matrix(state_se3)
    z_axis = R[:, 2]

    translated_points = np.zeros_like(points_3d)

    translated_points[..., Point3DIndex.X] = points_3d[..., Point3DIndex.X] + distance * z_axis[0]
    translated_points[..., Point3DIndex.Y] = points_3d[..., Point3DIndex.Y] + distance * z_axis[1]
    translated_points[..., Point3DIndex.Z] = points_3d[..., Point3DIndex.Z] + distance * z_axis[2]

    return translated_points
