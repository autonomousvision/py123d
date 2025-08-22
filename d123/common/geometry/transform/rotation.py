# import numpy as np
# import numpy.typing as npt

# from d123.common.geometry.base import Point3DIndex, StateSE3, StateSE3Index
# from d123.common.geometry.vector import Vector3D


# def get_roll_pitch_yaw_from_rotation_matrix(
#     rotation_matrix: npt.NDArray[np.float64],
# ) -> Vector3D:
#     """Extract roll, pitch, and yaw angles from a rotation matrix."""
#     assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3."

#     sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

#     singular = sy < 1e-6

#     if not singular:
#         x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
#         y = np.arctan2(-rotation_matrix[2, 0], sy)
#         z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
#     else:
#         x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
#         y = np.arctan2(-rotation_matrix[2, 0], sy)
#         z = 0.0

#     return Vector3D(x=x, y=y, z=z)
