# TODO: Properly implement and test these functions

# from typing import Union

# import numpy as np
# import numpy.typing as npt

# from d123.geometry import Vector3D
# from d123.geometry.geometry_index import Point3DIndex, QuaternionSE3Index, Vector3DIndex
# from d123.geometry.se import QuaternionSE3


# def translate_qse3_along_z(state_se3: QuaternionSE3, distance: float) -> QuaternionSE3:
#     """Translates a QuaternionSE3 state along the Z-axis.

#     :param state_se3: The QuaternionSE3 state to translate.
#     :param distance: The distance to translate along the Z-axis.
#     :return: The translated QuaternionSE3 state.
#     """
#     R = state_se3.rotation_matrix
#     z_axis = R[:, 2]

#     state_se3_array = state_se3.array.copy()
#     state_se3_array[QuaternionSE3Index.XYZ] += distance * z_axis[Vector3DIndex.XYZ]
#     return QuaternionSE3.from_array(state_se3_array, copy=False)


# def translate_qse3_along_y(state_se3: QuaternionSE3, distance: float) -> QuaternionSE3:
#     """Translates a QuaternionSE3 state along the Y-axis.

#     :param state_se3: The QuaternionSE3 state to translate.
#     :param distance: The distance to translate along the Y-axis.
#     :return: The translated QuaternionSE3 state.
#     """
#     R = state_se3.rotation_matrix
#     y_axis = R[:, 1]

#     state_se3_array = state_se3.array.copy()
#     state_se3_array[QuaternionSE3Index.XYZ] += distance * y_axis[Vector3DIndex.XYZ]
#     return QuaternionSE3.from_array(state_se3_array, copy=False)


# def translate_qse3_along_x(state_se3: QuaternionSE3, distance: float) -> QuaternionSE3:
#     """Translates a QuaternionSE3 state along the X-axis.

#     :param state_se3: The QuaternionSE3 state to translate.
#     :param distance: The distance to translate along the X-axis.
#     :return: The translated QuaternionSE3 state.
#     """
#     R = state_se3.rotation_matrix
#     x_axis = R[:, 0]

#     state_se3_array = state_se3.array.copy()
#     state_se3_array[QuaternionSE3Index.XYZ] += distance * x_axis[Vector3DIndex.XYZ]
#     return QuaternionSE3.from_array(state_se3_array, copy=False)


# def translate_qse3_along_body_frame(state_se3: QuaternionSE3, vector_3d: Vector3D) -> QuaternionSE3:
#     """Translates a QuaternionSE3 state along a vector in the body frame.

#     :param state_se3: The QuaternionSE3 state to translate.
#     :param vector_3d: The vector to translate along in the body frame.
#     :return: The translated QuaternionSE3 state.
#     """
#     R = state_se3.rotation_matrix
#     world_translation = R @ vector_3d.array

#     state_se3_array = state_se3.array.copy()
#     state_se3_array[QuaternionSE3Index.XYZ] += world_translation
#     return QuaternionSE3.from_array(state_se3_array, copy=False)


# def convert_absolute_to_relative_qse3_array(
#     origin: Union[QuaternionSE3, npt.NDArray[np.float64]], se3_array: npt.NDArray[np.float64]
# ) -> npt.NDArray[np.float64]:
#     """Converts a QuaternionSE3 array from the absolute frame to the relative frame.

#     :param origin: The origin state in the absolute frame, as a QuaternionSE3 or np.ndarray [x,y,z,qw,qx,qy,qz].
#     :param se3_array: The QuaternionSE3 array in the absolute frame [N, 7].
#     :raises TypeError: If the origin is not a QuaternionSE3 or np.ndarray.
#     :return: The QuaternionSE3 array in the relative frame [N, 7].
#     """
#     if isinstance(origin, QuaternionSE3):
#         origin_ = origin
#     elif isinstance(origin, np.ndarray):
#         assert origin.ndim == 1 and origin.shape[-1] == len(QuaternionSE3Index)
#         origin_ = QuaternionSE3.from_array(origin)
#     else:
#         raise TypeError(f"Expected QuaternionSE3 or np.ndarray, got {type(origin)}")

#     assert se3_array.ndim >= 1
#     assert se3_array.shape[-1] == len(QuaternionSE3Index)

#     t_origin = origin_.point_3d.array
#     R_origin = origin_.rotation_matrix

#     # Extract absolute positions and quaternions
#     abs_quaternions = se3_array[..., QuaternionSE3Index.QUATERNION]
#     q_origin = origin_.quaternion

#     # Compute relative quaternions: q_rel = q_origin^-1 * q_abs
#     if abs_quaternions.ndim == 1:
#         rel_quaternions = _quaternion_multiply(_quaternion_multiply(q_origin), abs_quaternions)
#     else:
#         rel_quaternions = np.array([_quaternion_multiply(_quaternion_multiply(q_origin), q) for q in abs_quaternions])


#     # Prepare output array
#     rel_se3_array = np.zeros_like(se3_array)
#     rel_se3_array[..., QuaternionSE3Index.XYZ] = (se3_array[..., QuaternionSE3Index.XYZ] - t_origin) @ R_origin
#     rel_se3_array[..., QuaternionSE3Index.QUATERNION] = rel_quaternions

#     return rel_se3_array


# def convert_relative_to_absolute_qse3_array(
#     origin: Union[QuaternionSE3, npt.NDArray[np.float64]], se3_array: npt.NDArray[np.float64]
# ) -> npt.NDArray[np.float64]:
#     """Converts a QuaternionSE3 array from the relative frame to the absolute frame.

#     :param origin: The origin state in the absolute frame, as a QuaternionSE3 or np.ndarray [x,y,z,qw,qx,qy,qz].
#     :param se3_array: The QuaternionSE3 array in the relative frame [N, 7].
#     :raises TypeError: If the origin is not a QuaternionSE3 or np.ndarray.
#     :return: The QuaternionSE3 array in the absolute frame [N, 7].
#     """
#     if isinstance(origin, QuaternionSE3):
#         t_origin = origin.translation
#         q_origin = origin.quaternion
#         R_origin = origin.rotation_matrix
#     elif isinstance(origin, np.ndarray):
#         assert origin.ndim == 1 and origin.shape[-1] == 7
#         t_origin = origin[:3]
#         q_origin = origin[3:]
#         origin_quat_se3 = QuaternionSE3.from_array(origin)
#         R_origin = origin_quat_se3.rotation_matrix
#     else:
#         raise TypeError(f"Expected QuaternionSE3 or np.ndarray, got {type(origin)}")

#     assert se3_array.ndim >= 1
#     assert se3_array.shape[-1] == len(QuaternionSE3Index)

#     # Extract relative positions and quaternions
#     rel_positions = se3_array[..., QuaternionSE3Index.XYZ]
#     rel_quaternions = se3_array[..., QuaternionSE3Index.QUATERNION]

#     # Compute absolute positions: R_origin @ p_rel + t_origin
#     abs_positions = (R_origin @ rel_positions.T).T + t_origin

#     # Compute absolute quaternions: q_abs = q_origin * q_rel
#     if rel_quaternions.ndim == 1:
#         abs_quaternions = _quaternion_multiply(q_origin, rel_quaternions)
#     else:
#         abs_quaternions = np.array([_quaternion_multiply(q_origin, q) for q in rel_quaternions])

#     # Prepare output array
#     abs_se3_array = se3_array.copy()
#     abs_se3_array[..., :3] = abs_positions
#     abs_se3_array[..., 3:] = abs_quaternions

#     return abs_se3_array


# def convert_absolute_to_relative_points_q3d_array(
#     origin: Union[QuaternionSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
# ) -> npt.NDArray[np.float64]:
#     """Converts 3D points from the absolute frame to the relative frame.

#     :param origin: The origin state in the absolute frame, as a QuaternionSE3 or np.ndarray [x,y,z,qw,qx,qy,qz].
#     :param points_3d_array: The 3D points in the absolute frame [N, 3].
#     :raises TypeError: If the origin is not a QuaternionSE3 or np.ndarray.
#     :return: The 3D points in the relative frame [N, 3].
#     """
#     if isinstance(origin, QuaternionSE3):
#         t_origin = origin.point_3d.array
#         R_origin_inv = origin.rotation_matrix.T
#     elif isinstance(origin, np.ndarray):
#         assert origin.ndim == 1 and origin.shape[-1] == 7
#         t_origin = origin[:3]
#         origin_quat_se3 = QuaternionSE3.from_array(origin)
#         R_origin_inv = origin_quat_se3.rotation_matrix.T
#     else:
#         raise TypeError(f"Expected QuaternionSE3 or np.ndarray, got {type(origin)}")

#     assert points_3d_array.ndim >= 1
#     assert points_3d_array.shape[-1] == len(Point3DIndex)

#     # Transform points: R_origin^T @ (p_abs - t_origin)
#     relative_points = (points_3d_array - t_origin) @ R_origin_inv.T
#     return relative_points


# def convert_relative_to_absolute_points_q3d_array(
#     origin: Union[QuaternionSE3, npt.NDArray[np.float64]], points_3d_array: npt.NDArray[np.float64]
# ) -> npt.NDArray[np.float64]:
#     """Converts 3D points from the relative frame to the absolute frame.

#     :param origin: The origin state in the absolute frame, as a QuaternionSE3 or np.ndarray [x,y,z,qw,qx,qy,qz].
#     :param points_3d_array: The 3D points in the relative frame [N, 3].
#     :raises TypeError: If the origin is not a QuaternionSE3 or np.ndarray.
#     :return: The 3D points in the absolute frame [N, 3].
#     """
#     if isinstance(origin, QuaternionSE3):
#         t_origin = origin.point_3d.array
#         R_origin = origin.rotation_matrix
#     elif isinstance(origin, np.ndarray):
#         assert origin.ndim == 1 and origin.shape[-1] == len(QuaternionSE3Index)
#         t_origin = origin[QuaternionSE3Index.XYZ]
#         origin_quat_se3 = QuaternionSE3.from_array(origin)
#         R_origin = origin_quat_se3.rotation_matrix
#     else:
#         raise TypeError(f"Expected QuaternionSE3 or np.ndarray, got {type(origin)}")

#     assert points_3d_array.shape[-1] == 3

#     # Transform points: R_origin @ p_rel + t_origin
#     absolute_points = (R_origin @ points_3d_array.T).T + t_origin
#     return absolute_points


# def _quaternion_multiply(q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#     """Multiply two quaternions [w, x, y, z].

#     :param q1: First quaternion [w, x, y, z].
#     :param q2: Second quaternion [w, x, y, z].
#     :return: Product quaternion [w, x, y, z].
#     """
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2

#     return np.array(
#         [
#             w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
#             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
#             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
#             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
#         ]
#     )
