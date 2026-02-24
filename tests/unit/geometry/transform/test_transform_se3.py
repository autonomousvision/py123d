import numpy as np
import numpy.typing as npt
import pytest

from py123d.geometry import EulerAngles, Point3D, PoseSE3, PoseSE3Index
from py123d.geometry.transform.transform_se3 import (
    abs_to_rel_point_3d,
    abs_to_rel_points_3d_array,
    abs_to_rel_se3,
    abs_to_rel_se3_array,
    convert_absolute_to_relative_se3_array,
    reframe_point_3d,
    reframe_points_3d_array,
    reframe_se3,
    reframe_se3_array,
    rel_to_abs_point_3d,
    rel_to_abs_points_3d_array,
    rel_to_abs_se3,
    rel_to_abs_se3_array,
    translate_se3_along_body_frame,
    translate_se3_along_x,
    translate_se3_along_y,
    translate_se3_along_z,
)
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_euler_array


class TestTransformSE3:
    def setup_method(self):
        quat_se3_a = PoseSE3.from_R_t(
            EulerAngles(roll=np.deg2rad(90), pitch=0.0, yaw=0.0),
            np.array([1.0, 2.0, 3.0]),
        )
        quat_se3_b = PoseSE3.from_R_t(
            EulerAngles(roll=0.0, pitch=np.deg2rad(90), yaw=0.0),
            np.array([1.0, -2.0, 3.0]),
        )
        quat_se3_c = PoseSE3.from_R_t(
            EulerAngles(roll=0.0, pitch=0.0, yaw=np.deg2rad(90)),
            np.array([-1.0, 2.0, -3.0]),
        )

        self.quat_se3 = [quat_se3_a, quat_se3_b, quat_se3_c]

        self.max_pose_xyz = 100.0

    def _get_random_quat_se3_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate random SE3 poses in quaternion representation."""
        # Generate random euler angles, then convert to quaternions
        euler_angles = np.zeros((size, 3), dtype=np.float64)
        euler_angles[:, 0] = np.random.uniform(-np.pi, np.pi, size)  # roll
        euler_angles[:, 1] = np.random.uniform(-np.pi / 2, np.pi / 2, size)  # pitch
        euler_angles[:, 2] = np.random.uniform(-np.pi, np.pi, size)  # yaw

        quat_se3_array = np.zeros((size, len(PoseSE3Index)), dtype=np.float64)
        quat_se3_array[:, PoseSE3Index.XYZ] = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz, (size, 3))
        quat_se3_array[:, PoseSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(euler_angles)

        return quat_se3_array

    def test_abs_to_rel_points_3d_array(self):
        random_points_3d = np.random.rand(10, 3)
        for quat_se3 in self.quat_se3:
            rel_points = abs_to_rel_points_3d_array(quat_se3, random_points_3d)
            # Round-trip check
            abs_points = rel_to_abs_points_3d_array(quat_se3, rel_points)
            np.testing.assert_allclose(abs_points, random_points_3d, atol=1e-6)

    def test_abs_to_rel_se3_array(self):
        for quat_se3 in self.quat_se3:
            random_quat_se3_array = self._get_random_quat_se3_array(np.random.randint(1, 10))

            rel_se3_quat = abs_to_rel_se3_array(quat_se3, random_quat_se3_array)
            # Round-trip check
            abs_se3_quat = rel_to_abs_se3_array(quat_se3, rel_se3_quat)
            np.testing.assert_allclose(
                abs_se3_quat[..., PoseSE3Index.XYZ], random_quat_se3_array[..., PoseSE3Index.XYZ], atol=1e-6
            )

    def test_rel_to_abs_points_3d_array(self):
        random_points_3d = np.random.rand(10, 3)
        for quat_se3 in self.quat_se3:
            abs_points = rel_to_abs_points_3d_array(quat_se3, random_points_3d)
            # Round-trip check
            rel_points = abs_to_rel_points_3d_array(quat_se3, abs_points)
            np.testing.assert_allclose(rel_points, random_points_3d, atol=1e-6)

    def test_rel_to_abs_se3_array(self):
        for quat_se3 in self.quat_se3:
            random_quat_se3_array = self._get_random_quat_se3_array(np.random.randint(1, 10))

            abs_se3_quat = rel_to_abs_se3_array(quat_se3, random_quat_se3_array)
            # Round-trip check
            rel_se3_quat = abs_to_rel_se3_array(quat_se3, abs_se3_quat)
            np.testing.assert_allclose(
                rel_se3_quat[..., PoseSE3Index.XYZ], random_quat_se3_array[..., PoseSE3Index.XYZ], atol=1e-6
            )

    def test_reframe_se3_array(self):
        for _ in range(10):
            random_quat_se3_array = self._get_random_quat_se3_array(np.random.randint(1, 10))

            from_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            to_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            identity_se3_array = np.zeros(len(PoseSE3Index), dtype=np.float64)
            identity_se3_array[PoseSE3Index.QW] = 1.0
            identity_se3 = PoseSE3.from_array(identity_se3_array)

            # Check if consistent with absolute-relative-absolute conversion
            converted_se3_quat = reframe_se3_array(from_se3, to_se3, random_quat_se3_array)

            abs_from_se3_quat = rel_to_abs_se3_array(from_se3, random_quat_se3_array)
            rel_to_se3_quat = abs_to_rel_se3_array(to_se3, abs_from_se3_quat)

            np.testing.assert_allclose(
                converted_se3_quat[..., PoseSE3Index.XYZ],
                rel_to_se3_quat[..., PoseSE3Index.XYZ],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                converted_se3_quat[..., PoseSE3Index.QUATERNION],
                rel_to_se3_quat[..., PoseSE3Index.QUATERNION],
                atol=1e-6,
            )

            # Check if consistent with absolute conversion to identity origin
            absolute_se3_quat = reframe_se3_array(from_se3, identity_se3, random_quat_se3_array)
            np.testing.assert_allclose(
                absolute_se3_quat[..., PoseSE3Index.XYZ],
                abs_from_se3_quat[..., PoseSE3Index.XYZ],
                atol=1e-6,
            )

    def test_reframe_points_3d_array(self):
        random_points_3d = np.random.rand(10, 3)
        for _ in range(10):
            from_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            to_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            identity_se3_array = np.zeros(len(PoseSE3Index), dtype=np.float64)
            identity_se3_array[PoseSE3Index.QW] = 1.0
            identity_se3 = PoseSE3.from_array(identity_se3_array)

            # Check if consistent with absolute-relative-absolute conversion
            converted_points_quat = reframe_points_3d_array(from_se3, to_se3, random_points_3d)
            abs_from_se3_quat = rel_to_abs_points_3d_array(from_se3, random_points_3d)
            rel_to_se3_quat = abs_to_rel_points_3d_array(to_se3, abs_from_se3_quat)
            np.testing.assert_allclose(converted_points_quat, rel_to_se3_quat, atol=1e-6)

            # Check if consistent with se3 array conversion
            random_se3_poses = np.zeros((random_points_3d.shape[0], len(PoseSE3Index)), dtype=np.float64)
            random_se3_poses[:, PoseSE3Index.XYZ] = random_points_3d
            random_se3_poses[:, PoseSE3Index.QUATERNION] = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
            converted_se3_quat_poses = reframe_se3_array(from_se3, to_se3, random_se3_poses)
            np.testing.assert_allclose(
                converted_se3_quat_poses[:, PoseSE3Index.XYZ],
                converted_points_quat,
                atol=1e-6,
            )

            # Check if consistent with absolute conversion to identity origin
            absolute_se3_quat = reframe_points_3d_array(from_se3, identity_se3, random_points_3d)
            np.testing.assert_allclose(
                absolute_se3_quat[..., PoseSE3Index.XYZ],
                abs_from_se3_quat[..., PoseSE3Index.XYZ],
                atol=1e-6,
            )

    def test_translate_se3_along_x(self):
        for _ in range(10):
            distance = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz)
            for quat_se3 in self.quat_se3:
                translated_quat = translate_se3_along_x(quat_se3, distance)
                # Verify rotation is preserved
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)
                # Verify translation is along local x-axis
                R = quat_se3.rotation_matrix
                expected_pos = quat_se3.point_3d.array + distance * R[:, 0]
                np.testing.assert_allclose(translated_quat.point_3d.array, expected_pos, atol=1e-6)

    def test_translate_se3_along_y(self):
        for _ in range(10):
            distance = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz)
            for quat_se3 in self.quat_se3:
                translated_quat = translate_se3_along_y(quat_se3, distance)
                # Verify rotation is preserved
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)
                # Verify translation is along local y-axis
                R = quat_se3.rotation_matrix
                expected_pos = quat_se3.point_3d.array + distance * R[:, 1]
                np.testing.assert_allclose(translated_quat.point_3d.array, expected_pos, atol=1e-6)

    def test_translate_se3_along_z(self):
        for _ in range(10):
            distance = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz)
            for quat_se3 in self.quat_se3:
                translated_quat = translate_se3_along_z(quat_se3, distance)
                # Verify rotation is preserved
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)
                # Verify translation is along local z-axis
                R = quat_se3.rotation_matrix
                expected_pos = quat_se3.point_3d.array + distance * R[:, 2]
                np.testing.assert_allclose(translated_quat.point_3d.array, expected_pos, atol=1e-6)

    def test_translate_se3_along_body_frame(self):
        for _ in range(10):
            vector_3d = Point3D(
                x=np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz),
                y=np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz),
                z=np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz),
            )
            for quat_se3 in self.quat_se3:
                translated_quat = translate_se3_along_body_frame(quat_se3, vector_3d)
                # Verify rotation is preserved
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)
                # Verify translation
                R = quat_se3.rotation_matrix
                expected_pos = quat_se3.point_3d.array + R @ vector_3d.array
                np.testing.assert_allclose(translated_quat.point_3d.array, expected_pos, atol=1e-6)

    # ──────────────────────────────────────────────────────────────────────────
    # Tests for new typed single-item functions
    # ──────────────────────────────────────────────────────────────────────────

    def test_abs_to_rel_se3(self) -> None:
        """Tests typed abs_to_rel_se3 returns correct PoseSE3."""
        for quat_se3 in self.quat_se3:
            random_pose = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            result = abs_to_rel_se3(quat_se3, random_pose)
            assert isinstance(result, PoseSE3)
            expected = abs_to_rel_se3_array(quat_se3, random_pose.array)
            np.testing.assert_allclose(result.array, expected, atol=1e-6)

    def test_rel_to_abs_se3(self) -> None:
        """Tests typed rel_to_abs_se3 returns correct PoseSE3."""
        for quat_se3 in self.quat_se3:
            random_pose = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            result = rel_to_abs_se3(quat_se3, random_pose)
            assert isinstance(result, PoseSE3)
            expected = rel_to_abs_se3_array(quat_se3, random_pose.array)
            np.testing.assert_allclose(result.array, expected, atol=1e-6)

    def test_reframe_se3(self) -> None:
        """Tests typed reframe_se3 matches array version."""
        for _ in range(10):
            from_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            to_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            pose = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            result = reframe_se3(from_origin, to_origin, pose)
            assert isinstance(result, PoseSE3)
            expected = reframe_se3_array(from_origin, to_origin, pose.array[np.newaxis])[0]
            np.testing.assert_allclose(result.array, expected, atol=1e-6)

    def test_abs_to_rel_point_3d(self) -> None:
        """Tests typed abs_to_rel_point_3d returns correct Point3D."""
        for quat_se3 in self.quat_se3:
            point = Point3D(np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            result = abs_to_rel_point_3d(quat_se3, point)
            assert isinstance(result, Point3D)
            expected = abs_to_rel_points_3d_array(quat_se3, point.array)
            np.testing.assert_allclose(result.array, expected, atol=1e-6)

    def test_rel_to_abs_point_3d(self) -> None:
        """Tests typed rel_to_abs_point_3d returns correct Point3D."""
        for quat_se3 in self.quat_se3:
            point = Point3D(np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            result = rel_to_abs_point_3d(quat_se3, point)
            assert isinstance(result, Point3D)
            expected = rel_to_abs_points_3d_array(quat_se3, point.array)
            np.testing.assert_allclose(result.array, expected, atol=1e-6)

    def test_reframe_point_3d(self) -> None:
        """Tests typed reframe_point_3d matches array version."""
        for _ in range(10):
            from_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            to_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            point = Point3D(np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            result = reframe_point_3d(from_origin, to_origin, point)
            assert isinstance(result, Point3D)
            expected = reframe_points_3d_array(from_origin, to_origin, point.array[np.newaxis])[0]
            np.testing.assert_allclose(result.array, expected, atol=1e-6)

    def test_typed_se3_round_trip(self) -> None:
        """Tests round-trip: abs_to_rel_se3 -> rel_to_abs_se3 returns original."""
        for _ in range(10):
            origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            pose = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            rel = abs_to_rel_se3(origin, pose)
            recovered = rel_to_abs_se3(origin, rel)
            np.testing.assert_allclose(pose.array[PoseSE3Index.XYZ], recovered.array[PoseSE3Index.XYZ], atol=1e-6)

    def test_typed_point_3d_round_trip(self) -> None:
        """Tests round-trip: abs_to_rel_point_3d -> rel_to_abs_point_3d returns original."""
        for _ in range(10):
            origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            point = Point3D(np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            rel = abs_to_rel_point_3d(origin, point)
            recovered = rel_to_abs_point_3d(origin, rel)
            np.testing.assert_allclose(point.array, recovered.array, atol=1e-6)

    # ──────────────────────────────────────────────────────────────────────────
    # Deprecation warning tests
    # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # Tests for _matmul_points_3d small-array path
    # ──────────────────────────────────────────────────────────────────────────

    def test_matmul_points_3d_small_array(self) -> None:
        """Tests _matmul_points_3d with a small array to hit the np.dot path."""
        from py123d.geometry.transform.transform_se3 import _matmul_points_3d

        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        matrix = np.eye(3, dtype=np.float64)
        result = _matmul_points_3d(points, matrix)
        np.testing.assert_allclose(result, points, atol=1e-10)

        # With a rotation matrix
        angle = np.pi / 4
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        result_rot = _matmul_points_3d(points, R)
        expected = points @ R
        np.testing.assert_allclose(result_rot, expected, atol=1e-10)

    # ──────────────────────────────────────────────────────────────────────────
    # Tests for _extract_rotation_translation_pose_arrays
    # ──────────────────────────────────────────────────────────────────────────

    def test_extract_with_ndarray_origin(self) -> None:
        """Tests that _extract_rotation_translation_pose_arrays works with ndarray input."""
        from py123d.geometry.transform.transform_se3 import _extract_rotation_translation_pose_arrays

        pose_array = np.zeros(len(PoseSE3Index), dtype=np.float64)
        pose_array[PoseSE3Index.QW] = 1.0
        pose_array[PoseSE3Index.X] = 1.0
        pose_array[PoseSE3Index.Y] = 2.0
        pose_array[PoseSE3Index.Z] = 3.0

        rotation, translation, result_array = _extract_rotation_translation_pose_arrays(pose_array)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(translation, [1.0, 2.0, 3.0], atol=1e-10)
        np.testing.assert_array_equal(result_array, pose_array)

    def test_extract_type_error(self) -> None:
        """Tests that _extract_rotation_translation_pose_arrays raises TypeError for invalid input."""
        from py123d.geometry.transform.transform_se3 import _extract_rotation_translation_pose_arrays

        with pytest.raises(TypeError, match="Expected"):
            _extract_rotation_translation_pose_arrays("not_a_pose")

        with pytest.raises(TypeError, match="Expected"):
            _extract_rotation_translation_pose_arrays(42)

    # ──────────────────────────────────────────────────────────────────────────
    # Tests for einsum large-array paths
    # ──────────────────────────────────────────────────────────────────────────

    def test_abs_to_rel_points_3d_large_array(self) -> None:
        """Tests abs_to_rel_points_3d_array with large arrays to hit einsum path."""
        origin = self.quat_se3[0]
        # Create large array (>8000 points to trigger einsum path)
        large_points = np.random.rand(10000, 3).astype(np.float64)
        small_points = large_points[:10]

        # Both should produce consistent round-trip results
        rel_large = abs_to_rel_points_3d_array(origin, large_points)
        abs_large = rel_to_abs_points_3d_array(origin, rel_large)
        np.testing.assert_allclose(abs_large, large_points, atol=1e-6)

        # Verify small subset matches
        rel_small = abs_to_rel_points_3d_array(origin, small_points)
        np.testing.assert_allclose(rel_large[:10], rel_small, atol=1e-6)

    def test_abs_to_rel_points_3d_large_3d_array(self) -> None:
        """Tests abs_to_rel_points_3d_array with large 3D arrays to hit einsum reshape path."""
        origin = self.quat_se3[0]
        # 3D array that exceeds threshold when flattened: 100*100*3 = 30000 > 8000*3
        large_3d_points = np.random.rand(100, 100, 3).astype(np.float64)
        rel_3d = abs_to_rel_points_3d_array(origin, large_3d_points)
        abs_3d = rel_to_abs_points_3d_array(origin, rel_3d)
        np.testing.assert_allclose(abs_3d, large_3d_points, atol=1e-6)

    def test_abs_to_rel_se3_large_array(self) -> None:
        """Tests abs_to_rel_se3_array with large arrays to hit einsum path."""
        origin = self.quat_se3[0]
        large_se3 = self._get_random_quat_se3_array(10000)

        rel_se3 = abs_to_rel_se3_array(origin, large_se3)
        abs_se3 = rel_to_abs_se3_array(origin, rel_se3)
        np.testing.assert_allclose(abs_se3[..., PoseSE3Index.XYZ], large_se3[..., PoseSE3Index.XYZ], atol=1e-6)

    def test_rel_to_abs_points_3d_large_array(self) -> None:
        """Tests rel_to_abs_points_3d_array with large arrays to hit einsum path."""
        origin = self.quat_se3[0]
        large_points = np.random.rand(10000, 3).astype(np.float64)

        abs_points = rel_to_abs_points_3d_array(origin, large_points)
        rel_points = abs_to_rel_points_3d_array(origin, abs_points)
        np.testing.assert_allclose(rel_points, large_points, atol=1e-6)

    def test_rel_to_abs_se3_large_array(self) -> None:
        """Tests rel_to_abs_se3_array with large arrays to hit einsum path."""
        origin = self.quat_se3[0]
        large_se3 = self._get_random_quat_se3_array(10000)

        abs_se3 = rel_to_abs_se3_array(origin, large_se3)
        rel_se3 = abs_to_rel_se3_array(origin, abs_se3)
        np.testing.assert_allclose(rel_se3[..., PoseSE3Index.XYZ], large_se3[..., PoseSE3Index.XYZ], atol=1e-6)

    def test_reframe_se3_large_array(self) -> None:
        """Tests reframe_se3_array with large arrays to hit einsum path."""
        from_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
        to_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
        large_se3 = self._get_random_quat_se3_array(10000)

        result = reframe_se3_array(from_origin, to_origin, large_se3)

        # Verify against two-step conversion
        abs_from = rel_to_abs_se3_array(from_origin, large_se3)
        expected = abs_to_rel_se3_array(to_origin, abs_from)
        np.testing.assert_allclose(result[..., PoseSE3Index.XYZ], expected[..., PoseSE3Index.XYZ], atol=1e-6)

    def test_reframe_points_3d_large_array(self) -> None:
        """Tests reframe_points_3d_array with large arrays to hit einsum path."""
        from_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
        to_origin = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
        large_points = np.random.rand(10000, 3).astype(np.float64)

        result = reframe_points_3d_array(from_origin, to_origin, large_points)

        # Verify against two-step conversion
        abs_from = rel_to_abs_points_3d_array(from_origin, large_points)
        expected = abs_to_rel_points_3d_array(to_origin, abs_from)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    # ──────────────────────────────────────────────────────────────────────────
    # Deprecation warning tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_deprecated_alias_emits_warning(self) -> None:
        """Tests that old function names emit DeprecationWarning."""
        import warnings

        origin = PoseSE3.from_R_t(EulerAngles(roll=0.0, pitch=0.0, yaw=0.0), np.array([0.0, 0.0, 0.0]))
        se3_array = np.zeros((1, len(PoseSE3Index)), dtype=np.float64)
        se3_array[0, PoseSE3Index.QW] = 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            convert_absolute_to_relative_se3_array(origin, se3_array)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "abs_to_rel_se3_array" in str(w[0].message)

    def test_deprecated_convert_absolute_to_relative_points_3d_array(self) -> None:
        """Tests that convert_absolute_to_relative_points_3d_array emits DeprecationWarning."""
        import warnings

        from py123d.geometry.transform.transform_se3 import convert_absolute_to_relative_points_3d_array

        origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([1.0, 2.0, 3.0]))
        points = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_absolute_to_relative_points_3d_array(origin, points)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "abs_to_rel_points_3d_array" in str(w[0].message)
        expected = abs_to_rel_points_3d_array(origin, points)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_deprecated_convert_relative_to_absolute_points_3d_array(self) -> None:
        """Tests that convert_relative_to_absolute_points_3d_array emits DeprecationWarning."""
        import warnings

        from py123d.geometry.transform.transform_se3 import convert_relative_to_absolute_points_3d_array

        origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([1.0, 2.0, 3.0]))
        points = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_relative_to_absolute_points_3d_array(origin, points)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "rel_to_abs_points_3d_array" in str(w[0].message)
        expected = rel_to_abs_points_3d_array(origin, points)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_deprecated_convert_relative_to_absolute_se3_array(self) -> None:
        """Tests that convert_relative_to_absolute_se3_array emits DeprecationWarning."""
        import warnings

        from py123d.geometry.transform.transform_se3 import convert_relative_to_absolute_se3_array

        origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0]))
        se3_array = np.zeros((1, len(PoseSE3Index)), dtype=np.float64)
        se3_array[0, PoseSE3Index.QW] = 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_relative_to_absolute_se3_array(origin, se3_array)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "rel_to_abs_se3_array" in str(w[0].message)
        expected = rel_to_abs_se3_array(origin, se3_array)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_deprecated_convert_se3_array_between_origins(self) -> None:
        """Tests that convert_se3_array_between_origins emits DeprecationWarning."""
        import warnings

        from py123d.geometry.transform.transform_se3 import convert_se3_array_between_origins

        from_origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([1.0, 0.0, 0.0]))
        to_origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([0.0, 1.0, 0.0]))
        se3_array = np.zeros((1, len(PoseSE3Index)), dtype=np.float64)
        se3_array[0, PoseSE3Index.QW] = 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_se3_array_between_origins(from_origin, to_origin, se3_array)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "reframe_se3_array" in str(w[0].message)
        expected = reframe_se3_array(from_origin, to_origin, se3_array)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_deprecated_convert_points_3d_array_between_origins(self) -> None:
        """Tests that convert_points_3d_array_between_origins emits DeprecationWarning."""
        import warnings

        from py123d.geometry.transform.transform_se3 import convert_points_3d_array_between_origins

        from_origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([1.0, 0.0, 0.0]))
        to_origin = PoseSE3.from_R_t(EulerAngles(0.0, 0.0, 0.0), np.array([0.0, 1.0, 0.0]))
        points = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_points_3d_array_between_origins(from_origin, to_origin, points)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "reframe_points_3d_array" in str(w[0].message)
        expected = reframe_points_3d_array(from_origin, to_origin, points)
        np.testing.assert_allclose(result, expected, atol=1e-10)
