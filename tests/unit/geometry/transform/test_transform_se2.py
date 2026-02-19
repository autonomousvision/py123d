import numpy as np
import numpy.typing as npt

from py123d.geometry import Point2D, PoseSE2, PoseSE2Index, Vector2D
from py123d.geometry.transform import (
    abs_to_rel_point_2d,
    abs_to_rel_points_2d_array,
    abs_to_rel_se2,
    abs_to_rel_se2_array,
    convert_absolute_to_relative_se2_array,  # deprecated alias, kept for deprecation warning test
    reframe_point_2d,
    reframe_points_2d_array,
    reframe_se2,
    reframe_se2_array,
    rel_to_abs_point_2d,
    rel_to_abs_points_2d_array,
    rel_to_abs_se2,
    rel_to_abs_se2_array,
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
        result: npt.NDArray[np.float64] = abs_to_rel_se2_array(origin, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_se2_array_with_rotation(self) -> None:
        """Tests converting absolute SE2 poses to relative SE2 poses with 90 degree rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 0.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = abs_to_rel_se2_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_se2_array_identity(self) -> None:
        """Tests converting absolute SE2 poses to relative SE2 poses with identity transformation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        absolute_poses: npt.NDArray[np.float64] = np.array([[1.0, 2.0, np.pi / 4]], dtype=np.float64)
        result: npt.NDArray[np.float64] = abs_to_rel_se2_array(reference, absolute_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 2.0, np.pi / 4]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_se2_array(self) -> None:
        """Tests converting relative SE2 poses to absolute SE2 poses."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        relative_poses: npt.NDArray[np.float64] = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        result: npt.NDArray[np.float64] = rel_to_abs_se2_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_se2_array_with_rotation(self) -> None:
        """Tests converting relative SE2 poses to absolute SE2 poses with rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 0.0, np.pi / 2], dtype=np.float64))
        relative_poses: npt.NDArray[np.float64] = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = rel_to_abs_se2_array(reference, relative_poses)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_point_2d_array(self) -> None:
        """Tests converting absolute 2D points to relative 2D points."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = abs_to_rel_points_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_point_2d_array_with_rotation(self) -> None:
        """Tests converting absolute 2D points to relative 2D points with 90 degree rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = abs_to_rel_points_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_absolute_to_relative_point_2d_array_empty(self) -> None:
        """Tests converting an empty array of absolute 2D points to relative 2D points."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        absolute_points: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 2)
        result: npt.NDArray[np.float64] = abs_to_rel_points_2d_array(reference, absolute_points)
        expected: npt.NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 2)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_point_2d_array(self) -> None:
        """Tests converting relative 2D points to absolute 2D points."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 1.0, 0.0], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = rel_to_abs_points_2d_array(reference, relative_points)
        expected: npt.NDArray[np.float64] = np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_convert_relative_to_absolute_point_2d_array_with_rotation(self) -> None:
        """Tests converting relative 2D points to absolute 2D points with 90 degree rotation."""
        reference: PoseSE2 = PoseSE2.from_array(np.array([1.0, 0.0, np.pi / 2], dtype=np.float64))
        relative_points: npt.NDArray[np.float64] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        result: npt.NDArray[np.float64] = rel_to_abs_points_2d_array(reference, relative_points)
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
            converted_points_quat = reframe_points_2d_array(from_se2, to_se2, random_points_2d)
            abs_from_se2 = rel_to_abs_points_2d_array(from_se2, random_points_2d)
            rel_to_se2 = abs_to_rel_points_2d_array(to_se2, abs_from_se2)
            np.testing.assert_allclose(converted_points_quat, rel_to_se2, atol=1e-6)

            # Check if consistent with absolute conversion to identity origin
            absolute_se2 = reframe_points_2d_array(from_se2, identity_se2, random_points_2d)
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
            converted_se2 = reframe_se2_array(from_se2, to_se2, random_se2_array)

            abs_from_se2 = rel_to_abs_se2_array(from_se2, random_se2_array)
            rel_to_se2 = abs_to_rel_se2_array(to_se2, abs_from_se2)

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
            absolute_se2 = reframe_se2_array(from_se2, identity_se2, random_se2_array)
            np.testing.assert_allclose(
                absolute_se2[..., PoseSE2Index.XY],
                abs_from_se2[..., PoseSE2Index.XY],
                atol=1e-6,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Tests for new typed single-item functions
    # ──────────────────────────────────────────────────────────────────────────

    def test_abs_to_rel_se2(self) -> None:
        """Tests typed abs_to_rel_se2 returns correct PoseSE2."""
        origin = PoseSE2(1.0, 1.0, 0.0)
        pose = PoseSE2(2.0, 2.0, 0.0)
        result = abs_to_rel_se2(origin, pose)
        assert isinstance(result, PoseSE2)
        np.testing.assert_array_almost_equal(result.array, np.array([1.0, 1.0, 0.0]), decimal=self.decimal)

    def test_rel_to_abs_se2(self) -> None:
        """Tests typed rel_to_abs_se2 returns correct PoseSE2."""
        origin = PoseSE2(1.0, 1.0, 0.0)
        pose = PoseSE2(1.0, 1.0, 0.0)
        result = rel_to_abs_se2(origin, pose)
        assert isinstance(result, PoseSE2)
        np.testing.assert_array_almost_equal(result.array, np.array([2.0, 2.0, 0.0]), decimal=self.decimal)

    def test_reframe_se2(self) -> None:
        """Tests typed reframe_se2 matches array version."""
        for _ in range(10):
            from_origin = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            to_origin = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            pose = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            result = reframe_se2(from_origin, to_origin, pose)
            assert isinstance(result, PoseSE2)
            result_array = reframe_se2_array(from_origin, to_origin, pose.array[np.newaxis])[0]
            np.testing.assert_array_almost_equal(result.array, result_array, decimal=self.decimal)

    def test_abs_to_rel_point_2d(self) -> None:
        """Tests typed abs_to_rel_point_2d returns correct Point2D."""
        origin = PoseSE2(1.0, 1.0, 0.0)
        point = Point2D(2.0, 2.0)
        result = abs_to_rel_point_2d(origin, point)
        assert isinstance(result, Point2D)
        np.testing.assert_array_almost_equal(result.array, np.array([1.0, 1.0]), decimal=self.decimal)

    def test_rel_to_abs_point_2d(self) -> None:
        """Tests typed rel_to_abs_point_2d returns correct Point2D."""
        origin = PoseSE2(1.0, 1.0, 0.0)
        point = Point2D(1.0, 1.0)
        result = rel_to_abs_point_2d(origin, point)
        assert isinstance(result, Point2D)
        np.testing.assert_array_almost_equal(result.array, np.array([2.0, 2.0]), decimal=self.decimal)

    def test_reframe_point_2d(self) -> None:
        """Tests typed reframe_point_2d matches array version."""
        for _ in range(10):
            from_origin = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            to_origin = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            point = Point2D(np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            result = reframe_point_2d(from_origin, to_origin, point)
            assert isinstance(result, Point2D)
            result_array = reframe_points_2d_array(from_origin, to_origin, point.array[np.newaxis])[0]
            np.testing.assert_array_almost_equal(result.array, result_array, decimal=self.decimal)

    def test_typed_se2_round_trip(self) -> None:
        """Tests round-trip: abs_to_rel_se2 -> rel_to_abs_se2 returns original."""
        for _ in range(10):
            origin = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            pose = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            rel = abs_to_rel_se2(origin, pose)
            recovered = rel_to_abs_se2(origin, rel)
            np.testing.assert_array_almost_equal(pose.array, recovered.array, decimal=self.decimal)

    def test_typed_point_2d_round_trip(self) -> None:
        """Tests round-trip: abs_to_rel_point_2d -> rel_to_abs_point_2d returns original."""
        for _ in range(10):
            origin = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            point = Point2D(np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            rel = abs_to_rel_point_2d(origin, point)
            recovered = rel_to_abs_point_2d(origin, rel)
            np.testing.assert_array_almost_equal(point.array, recovered.array, decimal=self.decimal)

    # ──────────────────────────────────────────────────────────────────────────
    # Tests for renamed array functions
    # ──────────────────────────────────────────────────────────────────────────

    def test_abs_to_rel_se2_array(self) -> None:
        """Tests abs_to_rel_se2_array matches deprecated convert_absolute_to_relative_se2_array."""
        origin = PoseSE2(1.0, 1.0, 0.0)
        poses = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        result = abs_to_rel_se2_array(origin, poses)
        expected = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_rel_to_abs_se2_array(self) -> None:
        """Tests rel_to_abs_se2_array matches deprecated convert_relative_to_absolute_se2_array."""
        origin = PoseSE2(1.0, 1.0, 0.0)
        poses = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, np.pi / 2]], dtype=np.float64)
        result = rel_to_abs_se2_array(origin, poses)
        expected = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, np.pi / 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_abs_to_rel_points_2d_array(self) -> None:
        """Tests abs_to_rel_points_2d_array with rotation."""
        origin = PoseSE2(0.0, 0.0, np.pi / 2)
        points = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        result = abs_to_rel_points_2d_array(origin, points)
        expected = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    def test_rel_to_abs_points_2d_array(self) -> None:
        """Tests rel_to_abs_points_2d_array with rotation."""
        origin = PoseSE2(1.0, 0.0, np.pi / 2)
        points = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        result = rel_to_abs_points_2d_array(origin, points)
        expected = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected, decimal=self.decimal)

    # ──────────────────────────────────────────────────────────────────────────
    # Deprecation warning tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_deprecated_alias_emits_warning(self) -> None:
        """Tests that old function names emit DeprecationWarning."""
        import warnings

        origin = PoseSE2(0.0, 0.0, 0.0)
        poses = np.array([[1.0, 2.0, 0.5]], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            convert_absolute_to_relative_se2_array(origin, poses)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "abs_to_rel_se2_array" in str(w[0].message)
