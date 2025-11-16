import numpy as np
import numpy.typing as npt

from py123d.geometry import EulerStateSE3, Vector3D
from py123d.geometry.transform.transform_euler_se3 import (
    convert_absolute_to_relative_euler_se3_array,
    convert_absolute_to_relative_points_3d_array,
    convert_relative_to_absolute_euler_se3_array,
    convert_relative_to_absolute_points_3d_array,
    translate_euler_se3_along_body_frame,
    translate_euler_se3_along_x,
    translate_euler_se3_along_y,
    translate_euler_se3_along_z,
)


class TestTransformEulerSE3:
    def setup_method(self):
        self.decimal = 6  # Decimal places for np.testing.assert_array_almost_equal
        self.num_consistency_tests = 10  # Number of random test cases for consistency checks

    def test_translate_se3_along_x(self) -> None:
        """Tests translating a SE3 state along the body frame forward direction."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: EulerStateSE3 = translate_euler_se3_along_x(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_x_negative(self) -> None:
        """Tests translating a SE3 state along the body frame backward direction."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = -0.5
        result: EulerStateSE3 = translate_euler_se3_along_x(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.5, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_x_with_rotation(self) -> None:
        """Tests translating a SE3 state along the body frame forward direction with yaw rotation."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64))
        distance: float = 2.5
        result: EulerStateSE3 = translate_euler_se3_along_x(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([0.0, 2.5, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_y(self) -> None:
        """Tests translating a SE3 state along the body frame lateral direction."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: EulerStateSE3 = translate_euler_se3_along_y(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_y_with_existing_position(self) -> None:
        """Tests translating a SE3 state along the body frame lateral direction with existing position."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 2.5
        result: EulerStateSE3 = translate_euler_se3_along_y(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 4.5, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_y_negative(self) -> None:
        """Tests translating a SE3 state along the body frame lateral direction in the negative direction."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = -1.0
        result: EulerStateSE3 = translate_euler_se3_along_y(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 1.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_y_with_rotation(self) -> None:
        """Tests translating a SE3 state along the body frame lateral direction with roll rotation."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, np.pi / 2, 0.0, 0.0], dtype=np.float64))
        distance: float = -1.0
        result: EulerStateSE3 = translate_euler_se3_along_y(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([1.0, 2.0, 2.0, np.pi / 2, 0.0, 0.0], dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_z(self) -> None:
        """Tests translating a SE3 state along the body frame vertical direction."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: EulerStateSE3 = translate_euler_se3_along_z(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_z_large_distance(self) -> None:
        """Tests translating a SE3 state along the body frame vertical direction with a large distance."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 10.0
        result: EulerStateSE3 = translate_euler_se3_along_z(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 15.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_z_negative(self) -> None:
        """Tests translating a SE3 state along the body frame vertical direction in the negative direction."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = -2.0
        result: EulerStateSE3 = translate_euler_se3_along_z(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_z_with_rotation(self) -> None:
        """Tests translating a SE3 state along the body frame vertical direction with pitch rotation."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, np.pi / 2, 0.0], dtype=np.float64))
        distance: float = 2.0
        result: EulerStateSE3 = translate_euler_se3_along_z(pose, distance)
        expected: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([3.0, 2.0, 3.0, 0.0, np.pi / 2, 0.0], dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_body_frame(self) -> None:
        """Tests translating a SE3 state along the body frame forward direction."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        result: EulerStateSE3 = translate_euler_se3_along_body_frame(pose, translation)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_body_frame_multiple_axes(self) -> None:
        """Tests translating a SE3 state along the body frame in multiple axes."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([0.5, -1.0, 2.0], dtype=np.float64))
        result: EulerStateSE3 = translate_euler_se3_along_body_frame(pose, translation)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.5, 1.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_body_frame_zero_translation(self) -> None:
        """Tests translating a SE3 state along the body frame with zero translation."""
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        result: EulerStateSE3 = translate_euler_se3_along_body_frame(pose, translation)
        expected: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array)

    def test_translate_se3_along_body_frame_with_rotation(self) -> None:
        """Tests translating a SE3 state along the body frame forward direction with yaw rotation."""
        # Rotate 90 degrees around z-axis, then translate 1 unit along body x-axis
        pose: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64))
        translation: Vector3D = Vector3D.from_array(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        result: EulerStateSE3 = translate_euler_se3_along_body_frame(pose, translation)
        # Should move in +Y direction in world frame
        expected: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se3_along_body_frame_consistency(self) -> None:
        """Tests consistency between translate_se3_along_body_frame and axis-specific translation functions."""

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

            original_pose: EulerStateSE3 = EulerStateSE3.from_array(
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
            result_body_frame_x: EulerStateSE3 = translate_euler_se3_along_body_frame(original_pose, translation_x)
            result_axis_x: EulerStateSE3 = translate_euler_se3_along_x(original_pose, x_distance)
            np.testing.assert_array_almost_equal(result_body_frame_x.array, result_axis_x.array, decimal=self.decimal)

            # y-axis translation
            translation_y: Vector3D = Vector3D.from_array(np.array([0.0, y_distance, 0.0], dtype=np.float64))
            result_body_frame_y: EulerStateSE3 = translate_euler_se3_along_body_frame(original_pose, translation_y)
            result_axis_y: EulerStateSE3 = translate_euler_se3_along_y(original_pose, y_distance)
            np.testing.assert_array_almost_equal(result_body_frame_y.array, result_axis_y.array, decimal=self.decimal)

            # z-axis translation
            translation_z: Vector3D = Vector3D.from_array(np.array([0.0, 0.0, z_distance], dtype=np.float64))
            result_body_frame_z: EulerStateSE3 = translate_euler_se3_along_body_frame(original_pose, translation_z)
            result_axis_z: EulerStateSE3 = translate_euler_se3_along_z(original_pose, z_distance)
            np.testing.assert_array_almost_equal(result_body_frame_z.array, result_axis_z.array, decimal=self.decimal)

            # all axes translation
            translation_all: Vector3D = Vector3D.from_array(
                np.array([x_distance, y_distance, z_distance], dtype=np.float64)
            )
            result_body_frame_all: EulerStateSE3 = translate_euler_se3_along_body_frame(original_pose, translation_all)
            intermediate_pose: EulerStateSE3 = translate_euler_se3_along_x(original_pose, x_distance)
            intermediate_pose = translate_euler_se3_along_y(intermediate_pose, y_distance)
            result_axis_all: EulerStateSE3 = translate_euler_se3_along_z(intermediate_pose, z_distance)
            np.testing.assert_array_almost_equal(
                result_body_frame_all.array, result_axis_all.array, decimal=self.decimal
            )

    def test_convert_absolute_to_relative_se3_array(self) -> None:
        """Tests converting absolute SE3 poses to relative SE3 poses."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array(
            [
                [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_euler_se3_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_se3_array_single_pose(self) -> None:
        """Tests converting a single absolute SE3 pose to a relative SE3 pose."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_euler_se3_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_se3_array_with_rotation(self) -> None:
        """Tests converting absolute SE3 poses to relative SE3 poses with 90 degree yaw rotation."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64)
        )
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_euler_se3_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[0.0, -1.0, 0.0, 0.0, 0.0, -np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_se3_array(self) -> None:
        """Tests converting relative SE3 poses to absolute SE3 poses."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        relative_poses: npt.NDArray[np.float64] = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_euler_se3_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array(
            [
                [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_se3_array_with_rotation(self) -> None:
        """Tests converting relative SE3 poses to absolute SE3 poses with 90 degree yaw rotation."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64)
        )
        relative_poses: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_euler_se3_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_points_3d_array(self) -> None:
        """Tests converting absolute 3D points to relative 3D points."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_3d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, -1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_points_3d_array_origin_reference(self) -> None:
        """Tests converting absolute 3D points to relative 3D points with origin reference."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_3d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_absolute_to_relative_points_3d_array_with_rotation(self) -> None:
        """Tests converting absolute 3D points to relative 3D points with 90 degree yaw rotation."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64)
        )
        absolute_points: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_3d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_points_3d_array(self) -> None:
        """Tests converting relative 3D points to absolute 3D points."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, -1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_3d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_points_3d_array_empty(self) -> None:
        """Tests converting an empty array of relative 3D points to absolute 3D points."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 3)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_3d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_relative_to_absolute_points_3d_array_with_rotation(self) -> None:
        """Tests converting relative 3D points to absolute 3D points with 90 degree yaw rotation."""
        reference: EulerStateSE3 = EulerStateSE3.from_array(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64)
        )
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_3d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)
