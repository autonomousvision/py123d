import numpy as np
import numpy.typing as npt

from py123d.geometry import PoseSE2, PoseSE2Index, Vector2D
from py123d.geometry.transform import (
    convert_absolute_to_relative_points_2d_array,
    convert_absolute_to_relative_se2_array,
    convert_points_2d_array_between_origins,
    convert_relative_to_absolute_points_2d_array,
    convert_relative_to_absolute_se2_array,
    convert_se2_array_between_origins,
    translate_se2_along_body_frame,
    translate_se2_along_x,
    translate_se2_along_y,
    translate_se2_array_along_body_frame,
)


class TestTransformSE2:
    def setup_method(self):
        self.decimal = 6  # Decimal places for np.testing.assert_array_almost_equal

    def _get_random_se2_array(self, num_poses: int) -> npt.NDArray[np.float64]:
        """Generates a random SE2 array for testing."""
        x = np.random.uniform(-10.0, 10.0, size=(num_poses,))
        y = np.random.uniform(-10.0, 10.0, size=(num_poses,))
        yaw = np.random.uniform(-np.pi, np.pi, size=(num_poses,))
        se2_array = np.stack((x, y, yaw), axis=-1)
        return se2_array

    def test_translate_se2_along_x(self) -> None:
        """Tests translating a SE2 state along the X-axis."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: PoseSE2 = translate_se2_along_x(pose, distance)
        expected: PoseSE2 = PoseSE2.from_array(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_x_negative(self) -> None:
        """Tests translating a SE2 state along the X-axis in the negative direction."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([1.0, 2.0, 0.0], dtype=np.float64))
        distance: float = -0.5
        result: PoseSE2 = translate_se2_along_x(pose, distance)
        expected: PoseSE2 = PoseSE2.from_array(np.array([0.5, 2.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_x_with_rotation(self) -> None:
        """Tests translating a SE2 state along the X-axis with 90 degree rotation."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        distance: float = 1.0
        result: PoseSE2 = translate_se2_along_x(pose, distance)
        expected: PoseSE2 = PoseSE2.from_array(np.array([0.0, 1.0, np.pi / 2], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_y(self) -> None:
        """Tests translating a SE2 state along the Y-axis."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        distance: float = 1.0
        result: PoseSE2 = translate_se2_along_y(pose, distance)
        expected: PoseSE2 = PoseSE2.from_array(np.array([0.0, 1.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_y_negative(self) -> None:
        """Tests translating a SE2 state along the Y-axis in the negative direction."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([1.0, 2.0, 0.0], dtype=np.float64))
        distance: float = -1.5
        result: PoseSE2 = translate_se2_along_y(pose, distance)
        expected: PoseSE2 = PoseSE2.from_array(np.array([1.0, 0.5, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_y_with_rotation(self) -> None:
        """Tests translating a SE2 state along the Y-axis with -90 degree rotation."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, -np.pi / 2], dtype=np.float64))
        distance: float = 2.0
        result: PoseSE2 = translate_se2_along_y(pose, distance)
        expected: PoseSE2 = PoseSE2.from_array(np.array([2.0, 0.0, -np.pi / 2], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_body_frame_forward(self) -> None:
        """Tests translating a SE2 state along the body frame forward direction, with 90 degree rotation."""
        # Move 1 unit forward in the direction of yaw (pi/2 = 90 degrees = +Y direction)
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        vector: Vector2D = Vector2D(1.0, 0.0)
        result: PoseSE2 = translate_se2_along_body_frame(pose, vector)
        expected: PoseSE2 = PoseSE2.from_array(np.array([0.0, 1.0, np.pi / 2], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_body_frame_backward(self) -> None:
        """Tests translating a SE2 state along the body frame backward direction."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        vector: Vector2D = Vector2D(-1.0, 0.0)
        result: PoseSE2 = translate_se2_along_body_frame(pose, vector)
        expected: PoseSE2 = PoseSE2.from_array(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_body_frame_diagonal(self) -> None:
        """Tests translating a SE2 state along the body frame diagonal direction."""
        pose: PoseSE2 = PoseSE2.from_array(np.array([1.0, 0.0, np.deg2rad(45)], dtype=np.float64))
        vector: Vector2D = Vector2D(1.0, 0.0)
        result: PoseSE2 = translate_se2_along_body_frame(pose, vector)
        expected: PoseSE2 = PoseSE2.from_array(
            np.array([1.0 + np.sqrt(2.0) / 2, 0.0 + np.sqrt(2.0) / 2, np.deg2rad(45)], dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_body_frame_lateral(self) -> None:
        """Tests translating a SE2 state along the body frame lateral direction."""
        # Move 1 unit to the right (positive y in body frame)
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        vector: Vector2D = Vector2D(0.0, 1.0)
        result: PoseSE2 = translate_se2_along_body_frame(pose, vector)
        expected: PoseSE2 = PoseSE2.from_array(np.array([0.0, 1.0, 0.0], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_along_body_frame_lateral_with_rotation(self) -> None:
        """Tests translating a SE2 state along the body frame lateral direction with 90 degree rotation."""
        # Move 1 unit to the right when facing 90 degrees
        pose: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        vector: Vector2D = Vector2D(0.0, 1.0)
        result: PoseSE2 = translate_se2_along_body_frame(pose, vector)
        expected: PoseSE2 = PoseSE2.from_array(np.array([-1.0, 0.0, np.pi / 2], dtype=np.float64))
        np.testing.assert_array_almost_equal(result.array, expected.array, decimal=self.decimal)

    def test_translate_se2_array_along_body_frame_single_distance(self) -> None:
        """Tests translating a SE2 state array along the body frame forward direction."""
        poses: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float64)
        distance: Vector2D = Vector2D(1.0, 0.0)
        result: npt.NDArray[np.float64] = translate_se2_array_along_body_frame(poses, distance)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_translate_se2_array_along_body_frame_multiple_distances(self) -> None:
        """Tests translating a SE2 state array along the body frame forward direction with different distances."""
        poses: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]], dtype=np.float64)
        distance: Vector2D = Vector2D(2.0, 0.0)
        result: npt.NDArray[np.float64] = translate_se2_array_along_body_frame(poses, distance)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, np.pi]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_translate_se2_array_along_body_frame_lateral(self) -> None:
        """Tests translating a SE2 state array along the body frame lateral direction with 90 degree rotation."""
        poses: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float64)
        distance: Vector2D = Vector2D(0.0, 1.0)
        result: npt.NDArray[np.float64] = translate_se2_array_along_body_frame(poses, distance)
        expected: npt.NDArray[np.float64] = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_se2_array(self) -> None:
        """Tests converting absolute SE2 poses to relative SE2 poses."""
        origin: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_se2_array(origin, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_se2_array_with_rotation(self) -> None:
        """Tests converting absolute SE2 poses to relative SE2 poses with 90 degree rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 0.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_se2_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_se2_array_identity(self) -> None:
        """Tests converting absolute SE2 poses to relative SE2 poses with identity transformation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 2.0, np.pi / 4]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_se2_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 2.0, np.pi / 4]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_se2_array(self) -> None:
        """Tests converting relative SE2 poses to absolute SE2 poses."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        relative_poses: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_se2_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_se2_array_with_rotation(self) -> None:
        """Tests converting relative SE2 poses to absolute SE2 poses with rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 0.0, np.pi / 2], dtype=np.float64))
        relative_poses: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_se2_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_point_2d_array(self) -> None:
        """Tests converting absolute 2D points to relative 2D points."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_point_2d_array_with_rotation(self) -> None:
        """Tests converting absolute 2D points to relative 2D points with 90 degree rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_point_2d_array_empty(self) -> None:
        """Tests converting an empty array of absolute 2D points to relative 2D points."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 2)
        result: npt.NDArray[np.float64] = convert_absolute_to_relative_points_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 2)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_point_2d_array(self) -> None:
        """Tests converting relative 2D points to absolute 2D points."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_2d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_point_2d_array_with_rotation(self) -> None:
        """Tests converting relative 2D points to absolute 2D points with 90 degree rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 0.0, np.pi / 2], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = convert_relative_to_absolute_points_2d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_points_2d_array_between_origins(self):
        random_points_2d = np.random.rand(10, 2)
        for _ in range(10):
            from_se2 = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            to_se2 = PoseSE2.from_array(self._get_random_se2_array(1)[0])

            identity_se2_array = np.zeros(len(PoseSE2Index), dtype=np.float64)
            identity_se2 = PoseSE2.from_array(identity_se2_array)

            # Check if consistent with absolute-relative-absolute conversion
            converted_points_quat = convert_points_2d_array_between_origins(from_se2, to_se2, random_points_2d)
            abs_from_se2 = convert_relative_to_absolute_points_2d_array(from_se2, random_points_2d)
            rel_to_se2 = convert_absolute_to_relative_points_2d_array(to_se2, abs_from_se2)
            np.testing.assert_allclose(converted_points_quat, rel_to_se2, atol=1e-6)

            # Check if consistent with absolute conversion to identity origin
            absolute_se2 = convert_points_2d_array_between_origins(from_se2, identity_se2, random_points_2d)
            np.testing.assert_allclose(
                absolute_se2[..., PoseSE2Index.XY],
                abs_from_se2[..., PoseSE2Index.XY],
                atol=1e-6,
            )

    def test_convert_se2_array_between_origins(self):
        for _ in range(10):
            random_se2_array = self._get_random_se2_array(np.random.randint(1, 10))

            from_se2 = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            to_se2 = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            identity_se2_array = np.zeros(len(PoseSE2Index), dtype=np.float64)
            identity_se2 = PoseSE2.from_array(identity_se2_array)

            # Check if consistent with absolute-relative-absolute conversion
            converted_se2 = convert_se2_array_between_origins(from_se2, to_se2, random_se2_array)

            abs_from_se2 = convert_relative_to_absolute_se2_array(from_se2, random_se2_array)
            rel_to_se2 = convert_absolute_to_relative_se2_array(to_se2, abs_from_se2)

            np.testing.assert_allclose(
                converted_se2[..., PoseSE2Index.XY],
                rel_to_se2[..., PoseSE2Index.XY],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                converted_se2[..., PoseSE2Index.YAW],
                rel_to_se2[..., PoseSE2Index.YAW],
                atol=1e-6,
            )

            # Check if consistent with absolute conversion to identity origin
            absolute_se2 = convert_se2_array_between_origins(from_se2, identity_se2, random_se2_array)
            np.testing.assert_allclose(
                absolute_se2[..., PoseSE2Index.XY],
                abs_from_se2[..., PoseSE2Index.XY],
                atol=1e-6,
            )
