import numpy as np
import numpy.typing as npt

from asim.common.geometry.base import StateSE3
from asim.common.geometry.vector import Vector3D


def get_rotation_matrix(state_se3: StateSE3) -> npt.NDArray[np.float64]:
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
    return R_z @ R_y @ R_x


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
