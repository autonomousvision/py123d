import unittest

import numpy as np
import numpy.typing as npt

from d123.geometry.geometry_index import EulerAnglesIndex, Point2DIndex, Point3DIndex, StateSE2Index, StateSE3Index
from d123.geometry.se import StateSE2, StateSE3
from d123.geometry.transform.transform_se2 import (
    convert_absolute_to_relative_point_2d_array,
    convert_absolute_to_relative_se2_array,
    convert_relative_to_absolute_point_2d_array,
    convert_relative_to_absolute_se2_array,
    translate_se2,
    translate_se2_along_body_frame,
    translate_se2_array,
    translate_se2_array_along_body_frame,
)
from d123.geometry.transform.transform_se3 import (
    convert_absolute_to_relative_points_3d_array,
    convert_absolute_to_relative_se3_array,
    convert_relative_to_absolute_points_3d_array,
    convert_relative_to_absolute_se3_array,
    translate_se3_along_body_frame,
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
        result: StateSE2 = translate_se2_along_body_frame(pose, vector)
        expected: StateSE2 = StateSE2.from_array(np.array([0.0, 1.0, np.deg2rad(90)], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_yaw_backward(self) -> None:
        pose: StateSE2 = StateSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        vector: Vector2D = Vector2D(-1.0, 0.0)
        result: StateSE2 = translate_se2_along_body_frame(pose, vector)
        expected: StateSE2 = StateSE2.from_array(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_yaw_diagonal(self) -> None:
        pose: StateSE2 = StateSE2.from_array(np.array([1.0, 0.0, np.deg2rad(45)], dtype=np.float64))
        vector: Vector2D = Vector2D(1.0, 0.0)
        result: StateSE2 = translate_se2_along_body_frame(pose, vector)
        expected: StateSE2 = StateSE2.from_array(
            np.array([1.0 + np.sqrt(2.0) / 2, 0.0 + np.sqrt(2.0) / 2, np.deg2rad(45)], dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_array_along_yaw(self) -> None:
        poses: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float64)
        distance: float = Vector2D(1.0, 0.0)
        result: npt.NDArray[np.float64] = translate_se2_array_along_body_frame(poses, distance)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_se2_array_along_yaw_multiple_distances(self) -> None:
        poses: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]], dtype=np.float64)
        distance: float = Vector2D(2.0, 0.0)
        result: npt.NDArray[np.float64] = translate_se2_array_along_body_frame(poses, distance)
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

    def setUp(self):
        self.decimal = 6  # Decimal places for np.testing.assert_array_almost_equal
        self.num_consistency_tests = 10  # Number of random test cases for consistency checks

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
        result: StateSE3 = translate_se3_along_body_frame(pose, translation)
        expected: StateSE3 = StateSE3.from_array(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_body_frame_multiple_axes(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([0.5, -1.0, 2.0], dtype=np.float64))
        result: StateSE3 = translate_se3_along_body_frame(pose, translation)
        expected: StateSE3 = StateSE3.from_array(np.array([1.5, 1.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_body_frame_zero_translation(self) -> None:
        pose: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        result: StateSE3 = translate_se3_along_body_frame(pose, translation)
        expected: StateSE3 = StateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_body_frame_consistency(self) -> None:

        for _ in range(self.num_consistency_tests):
            # Generate random parameters
            x_distance: float = np.random.uniform(-10.0, 10.0)
            y_distance: float = np.random.uniform(-10.0, 10.0)
            z_distance: float = np.random.uniform(-10.0, 10.0)

            start_x: float = np.random.uniform(-5.0, 5.0)
            start_y: float = np.random.uniform(-5.0, 5.0)
            start_z: float = np.random.uniform(-5.0, 5.0)

            start_roll: float = np.random.uniform(-np.pi, np.pi)
            start_pitch: float = np.random.uniform(-np.pi, np.pi)
            start_yaw: float = np.random.uniform(-np.pi, np.pi)

            original_pose: StateSE3 = StateSE3.from_array(
                np.array(
                    [
                        start_x,
                        start_y,
                        start_z,
                        start_roll,
                        start_pitch,
                        start_yaw,
                    ],
                    dtype=np.float64,
                )
            )

            # x-axis translation
            translation_x: Vector3D = Vector3D.from_array(np.array([x_distance, 0.0, 0.0], dtype=np.float64))
            result_body_frame_x: StateSE3 = translate_se3_along_body_frame(original_pose, translation_x)
            result_axis_x: StateSE3 = translate_se3_along_x(original_pose, x_distance)
            np.testing.assert_array_almost_equal(result_body_frame_x.array, result_axis_x.array, decimal=self.decimal)

            # y-axis translation
            translation_y: Vector3D = Vector3D.from_array(np.array([0.0, y_distance, 0.0], dtype=np.float64))
            result_body_frame_y: StateSE3 = translate_se3_along_body_frame(original_pose, translation_y)
            result_axis_y: StateSE3 = translate_se3_along_y(original_pose, y_distance)
            np.testing.assert_array_almost_equal(result_body_frame_y.array, result_axis_y.array, decimal=self.decimal)

            # z-axis translation
            translation_z: Vector3D = Vector3D.from_array(np.array([0.0, 0.0, z_distance], dtype=np.float64))
            result_body_frame_z: StateSE3 = translate_se3_along_body_frame(original_pose, translation_z)
            result_axis_z: StateSE3 = translate_se3_along_z(original_pose, z_distance)
            np.testing.assert_array_almost_equal(result_body_frame_z.array, result_axis_z.array, decimal=self.decimal)

            # all axes translation
            translation_all: Vector3D = Vector3D.from_array(
                np.array([x_distance, y_distance, z_distance], dtype=np.float64)
            )
            result_body_frame_all: StateSE3 = translate_se3_along_body_frame(original_pose, translation_all)
            intermediate_pose: StateSE3 = translate_se3_along_x(original_pose, x_distance)
            intermediate_pose = translate_se3_along_y(intermediate_pose, y_distance)
            result_axis_all: StateSE3 = translate_se3_along_z(intermediate_pose, z_distance)
            np.testing.assert_array_almost_equal(
                result_body_frame_all.array, result_axis_all.array, decimal=self.decimal
            )

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


class TestTransformConsistency(unittest.TestCase):
    def setUp(self):
        self.decimal = 6  # Decimal places for np.testing.assert_array_almost_equal
        self.num_consistency_tests = 10  # Number of random test cases for consistency checks

        self.max_pose_xyz = 100.0
        self.min_random_poses = 1
        self.max_random_poses = 20

    def _get_random_se2_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate a random SE2 pose"""
        random_se2_array = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz, (size, len(StateSE2Index)))
        random_se2_array[:, StateSE2Index.YAW] = np.random.uniform(-np.pi, np.pi, size)  # yaw angles
        return random_se2_array

    def _get_random_se3_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate a random SE3 poses"""
        random_se3_array = np.zeros((size, len(StateSE3Index)), dtype=np.float64)
        random_se3_array[:, StateSE3Index.XYZ] = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz, (size, 3))
        random_se3_array[:, StateSE3Index.EULER_ANGLES] = np.random.uniform(
            -np.pi, np.pi, (size, len(EulerAnglesIndex))
        )
        return random_se3_array

    def test_se2_absolute_relative_conversion_consistency(self) -> None:
        """Test that converting absolute->relative->absolute returns original poses"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = StateSE2.from_array(self._get_random_se2_array(1)[0])

            # Generate random absolute poses
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_poses = self._get_random_se2_array(num_poses)

            # Convert absolute -> relative -> absolute
            relative_poses = convert_absolute_to_relative_se2_array(reference, absolute_poses)
            recovered_absolute = convert_relative_to_absolute_se2_array(reference, relative_poses)

            np.testing.assert_array_almost_equal(absolute_poses, recovered_absolute, decimal=self.decimal)

    def test_se2_points_absolute_relative_conversion_consistency(self) -> None:
        """Test that converting absolute->relative->absolute returns original points"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = StateSE2.from_array(self._get_random_se2_array(1)[0])

            # Generate random absolute points
            num_points = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_points = self._get_random_se2_array(num_points)[:, StateSE2Index.XY]

            # Convert absolute -> relative -> absolute
            relative_points = convert_absolute_to_relative_point_2d_array(reference, absolute_points)
            recovered_absolute = convert_relative_to_absolute_point_2d_array(reference, relative_points)

            np.testing.assert_array_almost_equal(absolute_points, recovered_absolute, decimal=self.decimal)

    def test_se2_points_consistency(self) -> None:
        """Test whether SE2 point and pose conversions are consistent"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = StateSE2.from_array(self._get_random_se2_array(1)[0])

            # Generate random absolute points
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_se2 = self._get_random_se2_array(num_poses)

            # Convert absolute -> relative -> absolute
            relative_se2 = convert_absolute_to_relative_se2_array(reference, absolute_se2)
            relative_points = convert_absolute_to_relative_point_2d_array(
                reference, absolute_se2[..., StateSE2Index.XY]
            )
            np.testing.assert_array_almost_equal(
                relative_se2[..., StateSE2Index.XY], relative_points, decimal=self.decimal
            )

            recovered_absolute_se2 = convert_relative_to_absolute_se2_array(reference, relative_se2)
            absolute_points = convert_relative_to_absolute_point_2d_array(reference, relative_points)
            np.testing.assert_array_almost_equal(
                recovered_absolute_se2[..., StateSE2Index.XY], absolute_points, decimal=self.decimal
            )

    def test_se3_absolute_relative_conversion_consistency(self) -> None:
        """Test that converting absolute->relative->absolute returns original poses"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = StateSE3.from_array(self._get_random_se3_array(1)[0])

            # Generate random absolute poses
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_poses = self._get_random_se3_array(num_poses)

            # Convert absolute -> relative -> absolute
            relative_poses = convert_absolute_to_relative_se3_array(reference, absolute_poses)
            recovered_absolute = convert_relative_to_absolute_se3_array(reference, relative_poses)

            np.testing.assert_array_almost_equal(absolute_poses, recovered_absolute, decimal=self.decimal)

    def test_se3_points_absolute_relative_conversion_consistency(self) -> None:
        """Test that converting absolute->relative->absolute returns original points"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = StateSE3.from_array(self._get_random_se3_array(1)[0])

            # Generate random absolute points
            num_points = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_points = self._get_random_se3_array(num_points)[:, StateSE3Index.XYZ]

            # Convert absolute -> relative -> absolute
            relative_points = convert_absolute_to_relative_points_3d_array(reference, absolute_points)
            recovered_absolute = convert_relative_to_absolute_points_3d_array(reference, relative_points)

            np.testing.assert_array_almost_equal(absolute_points, recovered_absolute, decimal=self.decimal)

    def test_se3_points_consistency(self) -> None:
        """Test whether SE3 point and pose conversions are consistent"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = StateSE3.from_array(self._get_random_se3_array(1)[0])

            # Generate random absolute points
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_se3 = self._get_random_se3_array(num_poses)

            # Convert absolute -> relative -> absolute
            relative_se3 = convert_absolute_to_relative_se3_array(reference, absolute_se3)
            relative_points = convert_absolute_to_relative_points_3d_array(
                reference, absolute_se3[..., StateSE3Index.XYZ]
            )
            np.testing.assert_array_almost_equal(
                relative_se3[..., StateSE3Index.XYZ], relative_points, decimal=self.decimal
            )

            recovered_absolute_se3 = convert_relative_to_absolute_se3_array(reference, relative_se3)
            absolute_points = convert_relative_to_absolute_points_3d_array(reference, relative_points)
            np.testing.assert_array_almost_equal(
                recovered_absolute_se3[..., StateSE3Index.XYZ], absolute_points, decimal=self.decimal
            )

    def test_se2_se3_translation_along_body_consistency(self) -> None:
        """Test that SE2 and SE3 translations are consistent when SE3 has no z-component or rotation"""
        for _ in range(self.num_consistency_tests):
            # Create equivalent SE2 and SE3 poses (SE3 with z=0 and no rotations except yaw)

            pose_se2 = StateSE2.from_array(self._get_random_se2_array(1)[0])
            pose_se3 = StateSE3.from_array(
                np.array([pose_se2.x, pose_se2.y, 0.0, 0.0, 0.0, pose_se2.yaw], dtype=np.float64)
            )

            # Test translation along x-axis
            dx = np.random.uniform(-5.0, 5.0)
            translated_se2_x = translate_se2_along_body_frame(pose_se2, Vector2D(dx, 0.0))
            translated_se3_x = translate_se3_along_x(pose_se3, dx)

            np.testing.assert_array_almost_equal(
                translated_se2_x.array[StateSE2Index.XY], translated_se3_x.array[StateSE3Index.XY], decimal=self.decimal
            )
            np.testing.assert_almost_equal(
                translated_se2_x.array[StateSE2Index.YAW],
                translated_se3_x.array[StateSE3Index.YAW],
                decimal=self.decimal,
            )

            # Test translation along y-axis
            dy = np.random.uniform(-5.0, 5.0)
            translated_se2_y = translate_se2_along_body_frame(pose_se2, Vector2D(0.0, dy))
            translated_se3_y = translate_se3_along_y(pose_se3, dy)

            np.testing.assert_array_almost_equal(
                translated_se2_y.array[StateSE2Index.XY], translated_se3_y.array[StateSE3Index.XY], decimal=self.decimal
            )
            np.testing.assert_almost_equal(
                translated_se2_y.array[StateSE2Index.YAW],
                translated_se3_y.array[StateSE3Index.YAW],
                decimal=self.decimal,
            )

            # Test translation along x- and y-axis
            dx = np.random.uniform(-5.0, 5.0)
            dy = np.random.uniform(-5.0, 5.0)
            translated_se2_xy = translate_se2_along_body_frame(pose_se2, Vector2D(dx, dy))
            translated_se3_xy = translate_se3_along_body_frame(pose_se3, Vector3D(dx, dy, 0.0))
            np.testing.assert_array_almost_equal(
                translated_se2_xy.array[StateSE2Index.XY],
                translated_se3_xy.array[StateSE3Index.XY],
                decimal=self.decimal,
            )
            np.testing.assert_almost_equal(
                translated_se2_xy.array[StateSE2Index.YAW],
                translated_se3_xy.array[StateSE3Index.YAW],
                decimal=self.decimal,
            )

    def test_se2_se3_point_conversion_consistency(self) -> None:
        """Test that SE2 and SE3 point conversions are consistent for 2D points embedded in 3D"""
        for _ in range(self.num_consistency_tests):
            # Create equivalent SE2 and SE3 reference poses
            x = np.random.uniform(-10.0, 10.0)
            y = np.random.uniform(-10.0, 10.0)
            yaw = np.random.uniform(-np.pi, np.pi)

            reference_se2 = StateSE2.from_array(np.array([x, y, yaw], dtype=np.float64))
            reference_se3 = StateSE3.from_array(np.array([x, y, 0.0, 0.0, 0.0, yaw], dtype=np.float64))

            # Generate 2D points and embed them in 3D with z=0
            num_points = np.random.randint(1, 8)
            points_2d = np.random.uniform(-20.0, 20.0, (num_points, len(Point2DIndex)))
            points_3d = np.column_stack([points_2d, np.zeros(num_points)])

            # Convert using SE2 functions
            relative_2d = convert_absolute_to_relative_point_2d_array(reference_se2, points_2d)
            absolute_2d_recovered = convert_relative_to_absolute_point_2d_array(reference_se2, relative_2d)

            # Convert using SE3 functions
            relative_3d = convert_absolute_to_relative_points_3d_array(reference_se3, points_3d)
            absolute_3d_recovered = convert_relative_to_absolute_points_3d_array(reference_se3, relative_3d)

            # Check that SE2 and SE3 results are consistent (ignoring z-component)
            np.testing.assert_array_almost_equal(
                relative_2d,
                relative_3d[..., Point3DIndex.XY],
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(
                absolute_2d_recovered,
                absolute_3d_recovered[..., Point3DIndex.XY],
                decimal=self.decimal,
            )
            # Z-component should remain zero
            np.testing.assert_array_almost_equal(
                relative_3d[..., Point3DIndex.Z],
                np.zeros(num_points),
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(
                absolute_3d_recovered[..., Point3DIndex.Z],
                np.zeros(num_points),
                decimal=self.decimal,
            )


if __name__ == "__main__":
    unittest.main()
