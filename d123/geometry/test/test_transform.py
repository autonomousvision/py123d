# import unittest
# import numpy as np
# import numpy.typing as npt

# from d123.geometry.se import StateSE2, StateSE3
# from d123.geometry.transform.transform_se2 import (
#     convert_absolute_to_relative_se2_array,
#     convert_absolute_to_relative_point_2d_array,
#     convert_relative_to_absolute_se2_array,
#     convert_relative_to_absolute_point_2d_array,
#     translate_se2,
#     translate_se2_array,
#     translate_se2_along_yaw,
#     translate_se2_array_along_yaw,
# )
# from d123.geometry.transform.transform_se3 import (
#     translate_se3_along_z,
#     translate_se3_along_y,
#     translate_se3_along_x,
#     translate_body_frame,
#     convert_absolute_to_relative_se3_array,
#     convert_relative_to_absolute_se3_array,
#     convert_absolute_to_relative_points_3d_array,
#     convert_relative_to_absolute_points_3d_array,
# )
# from d123.geometry.vector import Vector2D


# class TestTransformSE2(unittest.TestCase):
#     def test_translate_se2(self) -> None:
#         pose: StateSE2 = StateSE2.from_array(np.array([1.0, 2.0, 0.0], dtype=float))
#         translation: Vector2D = Vector2D(1.0, 1.0)
#         result: StateSE2 = translate_se2(pose, translation)
#         expected: StateSE2 = StateSE2.from_array(np.array([2.0, 3.0, 0.0], dtype=float))
#         np.testing.assert_array_almost_equal(result.array, expected.array)

#     def test_translate_se2_array(self) -> None:
#         poses: npt.NDArray[np.float64] = np.array(
#             [[1.0, 2.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=float
#         )
#         translation: Vector2D = Vector2D(1.0, 1.0)
#         result: npt.NDArray[np.float64] = translate_se2_array(poses, translation)
#         expected: npt.NDArray[np.float64] = np.array(
#             [[2.0, 3.0, 0.0], [1.0, 1.0, np.pi / 2]], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_translate_se2_along_yaw(self) -> None:
#         # Move 1 unit forward in the direction of yaw (pi/2 = 90 degrees = +Y direction)
#         pose: npt.NDArray[np.float64] = np.array([0.0, 0.0, np.pi / 2], dtype=float)
#         distance: float = 1.0
#         result: npt.NDArray[np.float64] = translate_se2_along_yaw(pose, distance)
#         expected: npt.NDArray[np.float64] = np.array([0.0, 1.0, np.pi / 2], dtype=float)
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_translate_se2_array_along_yaw(self) -> None:
#         poses: npt.NDArray[np.float64] = np.array(
#             [[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=float
#         )
#         distance: float = 1.0
#         result: npt.NDArray[np.float64] = translate_se2_array_along_yaw(poses, distance)
#         expected: npt.NDArray[np.float64] = np.array(
#             [[1.0, 0.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_absolute_to_relative_se2_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array([1.0, 1.0, 0.0], dtype=float)
#         absolute_poses: npt.NDArray[np.float64] = np.array(
#             [[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=float
#         )
#         result: npt.NDArray[np.float64] = convert_absolute_to_relative_se2_array(
#             reference, absolute_poses
#         )
#         expected: npt.NDArray[np.float64] = np.array(
#             [[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_relative_to_absolute_se2_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array([1.0, 1.0, 0.0], dtype=float)
#         relative_poses: npt.NDArray[np.float64] = np.array(
#             [[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=float
#         )
#         result: npt.NDArray[np.float64] = convert_relative_to_absolute_se2_array(
#             reference, relative_poses
#         )
#         expected: npt.NDArray[np.float64] = np.array(
#             [[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_absolute_to_relative_point_2d_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array([1.0, 1.0, 0.0], dtype=float)
#         absolute_points: npt.NDArray[np.float64] = np.array(
#             [[2.0, 2.0], [0.0, 1.0]], dtype=float
#         )
#         result: npt.NDArray[np.float64] = convert_absolute_to_relative_point_2d_array(
#             reference, absolute_points
#         )
#         expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=float)
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_relative_to_absolute_point_2d_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array([1.0, 1.0, 0.0], dtype=float)
#         relative_points: npt.NDArray[np.float64] = np.array(
#             [[1.0, 1.0], [-1.0, 0.0]], dtype=float
#         )
#         result: npt.NDArray[np.float64] = convert_relative_to_absolute_point_2d_array(
#             reference, relative_points
#         )
#         expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=float)
#         np.testing.assert_array_almost_equal(result, expected)


# class TestTransformSE3(unittest.TestCase):
#     def test_translate_se3_along_x(self) -> None:
#         pose: npt.NDArray[np.float64] = np.array(
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         distance: float = 1.0
#         result: npt.NDArray[np.float64] = translate_se3_along_x(pose, distance)
#         expected: npt.NDArray[np.float64] = np.array(
#             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_translate_se3_along_y(self) -> None:
#         pose: npt.NDArray[np.float64] = np.array(
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         distance: float = 1.0
#         result: npt.NDArray[np.float64] = translate_se3_along_y(pose, distance)
#         expected: npt.NDArray[np.float64] = np.array(
#             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_translate_se3_along_z(self) -> None:
#         pose: npt.NDArray[np.float64] = np.array(
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         distance: float = 1.0
#         result: npt.NDArray[np.float64] = translate_se3_along_z(pose, distance)
#         expected: npt.NDArray[np.float64] = np.array(
#             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_translate_body_frame(self) -> None:
#         pose: npt.NDArray[np.float64] = np.array(
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         translation: npt.NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=float)
#         result: npt.NDArray[np.float64] = translate_body_frame(pose, translation)
#         expected: npt.NDArray[np.float64] = np.array(
#             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_absolute_to_relative_se3_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array(
#             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         absolute_poses: npt.NDArray[np.float64] = np.array(
#             [
#                 [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],
#                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#             ],
#             dtype=float,
#         )
#         result: npt.NDArray[np.float64] = convert_absolute_to_relative_se3_array(
#             reference, absolute_poses
#         )
#         expected: npt.NDArray[np.float64] = np.array(
#             [
#                 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
#                 [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
#             ],
#             dtype=float,
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_relative_to_absolute_se3_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array(
#             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         relative_poses: npt.NDArray[np.float64] = np.array(
#             [
#                 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
#                 [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
#             ],
#             dtype=float,
#         )
#         result: npt.NDArray[np.float64] = convert_relative_to_absolute_se3_array(
#             reference, relative_poses
#         )
#         expected: npt.NDArray[np.float64] = np.array(
#             [
#                 [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],
#                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#             ],
#             dtype=float,
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_absolute_to_relative_points_3d_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array(
#             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         absolute_points: npt.NDArray[np.float64] = np.array(
#             [[2.0, 2.0, 2.0], [0.0, 1.0, 0.0]], dtype=float
#         )
#         result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_3d_array(
#             reference, absolute_points
#         )
#         expected: npt.NDArray[np.float64] = np.array(
#             [[1.0, 1.0, 1.0], [-1.0, 0.0, -1.0]], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_convert_relative_to_absolute_points_3d_array(self) -> None:
#         reference: npt.NDArray[np.float64] = np.array(
#             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float
#         )
#         relative_points: npt.NDArray[np.float64] = np.array(
#             [[1.0, 1.0, 1.0], [-1.0, 0.0, -1.0]], dtype=float
#         )
#         result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_3d_array(
#             reference, relative_points
#         )
#         expected: npt.NDArray[np.float64] = np.array(
#             [[2.0, 2.0, 2.0], [0.0, 1.0, 0.0]], dtype=float
#         )
#         np.testing.assert_array_almost_equal(result, expected)


# if __name__ == "__main__":
#     unittest.main()
