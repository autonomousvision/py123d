import unittest

import numpy as np
import numpy.typing as npt

from d123.geometry.se import StateSE2, StateSE3
from d123.geometry.transform.transform_se2 import (
    convert_absolute_to_relative_point_2d_array,
    convert_absolute_to_relative_se2_array,
    convert_relative_to_absolute_point_2d_array,
    convert_relative_to_absolute_se2_array,
    translate_se2,
    translate_se2_along_yaw,
    translate_se2_array,
    translate_se2_array_along_yaw,
)
from d123.geometry.transform.transform_se3 import (
    convert_absolute_to_relative_points_3d_array,
    convert_absolute_to_relative_se3_array,
    convert_relative_to_absolute_points_3d_array,
    convert_relative_to_absolute_se3_array,
    translate_body_frame,
    translate_se3_along_x,
    translate_se3_along_y,
    translate_se3_along_z,
)
from d123.geometry.vector import Vector2D, Vector3D


class TestTransformSE2(unittest.TestCase):

    def setUp(self):
        self.decimal = 6  # Decimal places for np.testing.assert_array_almost_equal

    def test_translate_se2(self) -> None:
        pose: StateSE2 = StateSE2.from_array(np.array([1.0, 2.0, 0.0], dtype=np.float64))
        translation: Vector2D = Vector2D(1.0, 1.0)

        result: StateSE2 = translate_se2(pose, translation)
        expected: StateSE2 = StateSE2.from_array(np.array([2.0, 3.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_negative_translation(self) -> None:
        pose: StateSE2 = StateSE2.from_array(np.array([1.0, 2.0, 0.0], dtype=np.float64))
        translation: Vector2D = Vector2D(-0.5, -1.5)
        result: StateSE2 = translate_se2(pose, translation)
        expected: StateSE2 = StateSE2.from_array(np.array([0.5, 0.5, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_with_rotation(self) -> None:
        pose: StateSE2 = StateSE2.from_array(np.array([0.0, 0.0, np.pi / 4], dtype=np.float64))
        translation: Vector2D = Vector2D(1.0, 0.0)
        result: StateSE2 = translate_se2(pose, translation)
        expected: StateSE2 = StateSE2.from_array(np.array([1.0, 0.0, np.pi / 4], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_array(self) -> None:
        poses: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float64)
        translation: Vector2D = Vector2D(1.0, 1.0)
        result: npt.NDArray[np.float64] = translate_se2_array(poses, translation)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 3.0, 0.0], [1.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_se2_array_zero_translation(self) -> None:
        poses: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float64)
        translation: Vector2D = Vector2D(0.0, 0.0)
        result: npt.NDArray[np.float64] = translate_se2_array(poses, translation)
        expected: npt.NDArray[np.float64] = poses.copy()
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_se2_along_yaw(self) -> None:
        # Move 1 unit forward in the direction of yaw (pi/2 = 90 degrees = +Y direction)
        pose: StateSE2 = StateSE2.from_array(np.array([0.0, 0.0, np.deg2rad(90)], dtype=np.float64))
        vector: Vector2D = Vector2D(1.0, 0.0)
        result: StateSE2 = translate_se2_along_yaw(pose, vector)
        expected: StateSE2 = StateSE2.from_array(np.array([0.0, 1.0, np.deg2rad(90)], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_yaw_backward(self) -> None:
        pose: StateSE2 = StateSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        vector: Vector2D = Vector2D(-1.0, 0.0)
        result: StateSE2 = translate_se2_along_yaw(pose, vector)
        expected: StateSE2 = StateSE2.from_array(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_yaw_diagonal(self) -> None:
        pose: StateSE2 = StateSE2.from_array(np.array([1.0, 0.0, np.deg2rad(45)], dtype=np.float64))
        vector: Vector2D = Vector2D(1.0, 0.0)
        result: StateSE2 = translate_se2_along_yaw(pose, vector)
        expected: StateSE2 = StateSE2.from_array(
            np.array([1.0 + np.sqrt(2.0) / 2, 0.0 + np.sqrt(2.0) / 2, np.deg2rad(45)], dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_array_along_yaw(self) -> None:
        poses: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float64)
        distance: float = Vector2D(1.0, 0.0)
        result: npt.NDArray[np.float64] = translate_se2_array_along_yaw(poses, distance)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_se2_array_along_yaw_multiple_distances(self) -> None:
        poses: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]], dtype=np.float64)
        distance: float = Vector2D(2.0, 0.0)
        result: npt.NDArray[np.float64] = translate_se2_array_along_yaw(poses, distance)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, np.pi]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_se2_array(self) -> None:
        origin: StateSE2 = StateSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_se2_array(origin, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_se2_array_with_rotation(self) -> None:
        reference: StateSE2 = StateSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 0.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_se2_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_se2_array(self) -> None:
        reference: StateSE2 = StateSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        relative_poses: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_se2_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_point_2d_array(self) -> None:
        reference: StateSE2 = StateSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_point_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_point_2d_array_with_rotation(self) -> None:
        reference: StateSE2 = StateSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_point_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_point_2d_array(self) -> None:
        reference: StateSE2 = StateSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_point_2d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)


class TestTransformSE3(unittest.TestCase):
    def test_translate_se3_along_x(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: StateSE3 = translate_se3_along_x(pose, distance)
        expected: StateSE3 = StateSE3.from_array(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_x_negative(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = -0.5
        result: StateSE3 = translate_se3_along_x(pose, distance)
        expected: StateSE3 = StateSE3.from_array(np.array([0.5, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_y(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: StateSE3 = translate_se3_along_y(pose, distance)
        expected: StateSE3 = StateSE3.from_array(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_y_with_existing_position(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 2.5
        result: StateSE3 = translate_se3_along_y(pose, distance)
        expected: StateSE3 = StateSE3.from_array(np.array([1.0, 4.5, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_z(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: StateSE3 = translate_se3_along_z(pose, distance)
        expected: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_z_large_distance(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 10.0
        result: StateSE3 = translate_se3_along_z(pose, distance)
        expected: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 15.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_body_frame(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        result: StateSE3 = translate_body_frame(pose, translation)
        expected: StateSE3 = StateSE3.from_array(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_body_frame_multiple_axes(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([0.5, -1.0, 2.0], dtype=np.float64))
        result: StateSE3 = translate_body_frame(pose, translation)
        expected: StateSE3 = StateSE3.from_array(np.array([1.5, 1.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_body_frame_zero_translation(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        result: StateSE3 = translate_body_frame(pose, translation)
        expected: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_convert_absolute_to_relative_se3_array(self) -> None:
        reference: StateSE3 = StateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array(
            [
                [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_se3_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_se3_array_single_pose(self) -> None:
        reference: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_se3_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_se3_array(self) -> None:
        reference: StateSE3 = StateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        relative_poses: npt.NDArray[np.float64] = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_se3_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array(
            [
                [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_points_3d_array(self) -> None:
        reference: StateSE3 = StateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_3d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, -1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_points_3d_array_origin_reference(self) -> None:
        reference: StateSE3 = StateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_3d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_points_3d_array(self) -> None:
        reference: StateSE3 = StateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, -1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_3d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_points_3d_array_empty(self) -> None:
        reference: StateSE3 = StateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 3)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_3d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 3)
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
