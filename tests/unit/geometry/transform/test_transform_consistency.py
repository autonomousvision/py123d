import numpy as np
import numpy.typing as npt

from py123d.geometry import EulerPoseSE3, PoseSE2, Vector2D, Vector3D
from py123d.geometry.geometry_index import EulerPoseSE3Index, Point2DIndex, Point3DIndex, PoseSE2Index
from py123d.geometry.transform.transform_euler_se3 import (
    convert_absolute_to_relative_euler_se3_array,
    convert_absolute_to_relative_points_3d_array,
    convert_relative_to_absolute_euler_se3_array,
    convert_relative_to_absolute_points_3d_array,
    translate_euler_se3_along_body_frame,
    translate_euler_se3_along_x,
    translate_euler_se3_along_y,
)
from py123d.geometry.transform.transform_se2 import (
    convert_absolute_to_relative_points_2d_array,
    convert_absolute_to_relative_se2_array,
    convert_relative_to_absolute_points_2d_array,
    convert_relative_to_absolute_se2_array,
    translate_se2_along_body_frame,
    translate_se2_along_x,
    translate_se2_along_y,
    translate_se2_array_along_body_frame,
)
from py123d.geometry.utils.rotation_utils import get_rotation_matrices_from_euler_array


class TestTransformConsistency:
    """Tests to ensure consistency between different transformation functions."""

    def setup_method(self):
        self.decimal = 4  # Decimal places for np.testing.assert_array_almost_equal
        self.num_consistency_tests = 10  # Number of random test cases for consistency checks

        self.max_pose_xyz = 100.0
        self.min_random_poses = 1
        self.max_random_poses = 20

    def _get_random_se2_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate a random SE2 pose"""
        random_se2_array = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz, (size, len(PoseSE2Index)))
        random_se2_array[:, PoseSE2Index.YAW] = np.random.uniform(-np.pi, np.pi, size)  # yaw angles
        return random_se2_array

    def _get_random_se3_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate a random SE3 poses"""
        random_se3_array = np.zeros((size, len(EulerPoseSE3Index)), dtype=np.float64)
        random_se3_array[:, EulerPoseSE3Index.XYZ] = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz, (size, 3))
        random_se3_array[:, EulerPoseSE3Index.YAW] = np.random.uniform(-np.pi, np.pi, size)
        random_se3_array[:, EulerPoseSE3Index.PITCH] = np.random.uniform(-np.pi / 2, np.pi / 2, size)
        random_se3_array[:, EulerPoseSE3Index.ROLL] = np.random.uniform(-np.pi, np.pi, size)

        return random_se3_array

    def test_se2_absolute_relative_conversion_consistency(self) -> None:
        """Test that converting absolute->relative->absolute returns original poses"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = PoseSE2.from_array(self._get_random_se2_array(1)[0])

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
            reference = PoseSE2.from_array(self._get_random_se2_array(1)[0])

            # Generate random absolute points
            num_points = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_points = self._get_random_se2_array(num_points)[:, PoseSE2Index.XY]

            # Convert absolute -> relative -> absolute
            relative_points = convert_absolute_to_relative_points_2d_array(reference, absolute_points)
            recovered_absolute = convert_relative_to_absolute_points_2d_array(reference, relative_points)

            np.testing.assert_array_almost_equal(absolute_points, recovered_absolute, decimal=self.decimal)

    def test_se2_points_consistency(self) -> None:
        """Test whether SE2 point and pose conversions are consistent"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = PoseSE2.from_array(self._get_random_se2_array(1)[0])

            # Generate random absolute points
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_se2 = self._get_random_se2_array(num_poses)

            # Convert absolute -> relative -> absolute
            relative_se2 = convert_absolute_to_relative_se2_array(reference, absolute_se2)
            relative_points = convert_absolute_to_relative_points_2d_array(
                reference, absolute_se2[..., PoseSE2Index.XY]
            )
            np.testing.assert_array_almost_equal(
                relative_se2[..., PoseSE2Index.XY], relative_points, decimal=self.decimal
            )

            recovered_absolute_se2 = convert_relative_to_absolute_se2_array(reference, relative_se2)
            absolute_points = convert_relative_to_absolute_points_2d_array(reference, relative_points)
            np.testing.assert_array_almost_equal(
                recovered_absolute_se2[..., PoseSE2Index.XY], absolute_points, decimal=self.decimal
            )

    def test_se2_translation_consistency(self) -> None:
        """Test that SE2 translations are consistent between different methods"""
        for _ in range(self.num_consistency_tests):
            # Generate random pose
            pose = PoseSE2.from_array(self._get_random_se2_array(1)[0])

            # Generate random distances
            dx = np.random.uniform(-10.0, 10.0)
            dy = np.random.uniform(-10.0, 10.0)

            # Test x-translation consistency
            result_x_direct = translate_se2_along_x(pose, dx)
            result_x_body = translate_se2_along_body_frame(pose, Vector2D(dx, 0.0))
            np.testing.assert_array_almost_equal(result_x_direct.array, result_x_body.array, decimal=self.decimal)

            # Test y-translation consistency
            result_y_direct = translate_se2_along_y(pose, dy)
            result_y_body = translate_se2_along_body_frame(pose, Vector2D(0.0, dy))
            np.testing.assert_array_almost_equal(result_y_direct.array, result_y_body.array, decimal=self.decimal)

            # Test combined translation
            result_xy_body = translate_se2_along_body_frame(pose, Vector2D(dx, dy))
            result_xy_sequential = translate_se2_along_y(translate_se2_along_x(pose, dx), dy)
            np.testing.assert_array_almost_equal(result_xy_body.array, result_xy_sequential.array, decimal=self.decimal)

    def test_se3_absolute_relative_conversion_consistency(self) -> None:
        """Test that converting absolute->relative->absolute returns original poses"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = EulerPoseSE3.from_array(self._get_random_se3_array(1)[0])

            # Generate random absolute poses
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_poses = self._get_random_se3_array(num_poses)

            # Convert absolute -> relative -> absolute
            relative_poses = convert_absolute_to_relative_euler_se3_array(reference, absolute_poses)
            recovered_absolute = convert_relative_to_absolute_euler_se3_array(reference, relative_poses)

            np.testing.assert_array_almost_equal(
                absolute_poses[..., EulerPoseSE3Index.XYZ],
                recovered_absolute[..., EulerPoseSE3Index.XYZ],
                decimal=self.decimal,
            )

            absolute_rotation_matrices = get_rotation_matrices_from_euler_array(
                absolute_poses[..., EulerPoseSE3Index.EULER_ANGLES]
            )
            recovered_rotation_matrices = get_rotation_matrices_from_euler_array(
                recovered_absolute[..., EulerPoseSE3Index.EULER_ANGLES]
            )

            np.testing.assert_array_almost_equal(
                absolute_rotation_matrices,
                recovered_rotation_matrices,
                decimal=self.decimal,
            )

    def test_se3_points_absolute_relative_conversion_consistency(self) -> None:
        """Test that converting absolute->relative->absolute returns original points"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = EulerPoseSE3.from_array(self._get_random_se3_array(1)[0])

            # Generate random absolute points
            num_points = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_points = self._get_random_se3_array(num_points)[:, EulerPoseSE3Index.XYZ]

            # Convert absolute -> relative -> absolute
            relative_points = convert_absolute_to_relative_points_3d_array(reference, absolute_points)
            recovered_absolute = convert_relative_to_absolute_points_3d_array(reference, relative_points)

            np.testing.assert_array_almost_equal(absolute_points, recovered_absolute, decimal=self.decimal)

    def test_se3_points_consistency(self) -> None:
        """Test whether SE3 point and pose conversions are consistent"""
        for _ in range(self.num_consistency_tests):
            # Generate random reference pose
            reference = EulerPoseSE3.from_array(self._get_random_se3_array(1)[0])

            # Generate random absolute points
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            absolute_se3 = self._get_random_se3_array(num_poses)

            # Convert absolute -> relative -> absolute
            relative_se3 = convert_absolute_to_relative_euler_se3_array(reference, absolute_se3)
            relative_points = convert_absolute_to_relative_points_3d_array(
                reference, absolute_se3[..., EulerPoseSE3Index.XYZ]
            )
            np.testing.assert_array_almost_equal(
                relative_se3[..., EulerPoseSE3Index.XYZ], relative_points, decimal=self.decimal
            )

            recovered_absolute_se3 = convert_relative_to_absolute_euler_se3_array(reference, relative_se3)
            absolute_points = convert_relative_to_absolute_points_3d_array(reference, relative_points)
            np.testing.assert_array_almost_equal(
                recovered_absolute_se3[..., EulerPoseSE3Index.XYZ], absolute_points, decimal=self.decimal
            )

    def test_se2_se3_translation_along_body_consistency(self) -> None:
        """Test that SE2 and SE3 translations are consistent when SE3 has no z-component or rotation"""
        for _ in range(self.num_consistency_tests):
            # Create equivalent SE2 and SE3 poses (SE3 with z=0 and no rotations except yaw)

            pose_se2 = PoseSE2.from_array(self._get_random_se2_array(1)[0])
            pose_se3 = EulerPoseSE3.from_array(
                np.array([pose_se2.x, pose_se2.y, 0.0, 0.0, 0.0, pose_se2.yaw], dtype=np.float64)
            )

            # Test translation along x-axis
            dx = np.random.uniform(-5.0, 5.0)
            translated_se2_x = translate_se2_along_body_frame(pose_se2, Vector2D(dx, 0.0))
            translated_se3_x = translate_euler_se3_along_x(pose_se3, dx)

            np.testing.assert_array_almost_equal(
                translated_se2_x.array[PoseSE2Index.XY],
                translated_se3_x.array[EulerPoseSE3Index.XY],
                decimal=self.decimal,
            )
            np.testing.assert_almost_equal(
                translated_se2_x.array[PoseSE2Index.YAW],
                translated_se3_x.array[EulerPoseSE3Index.YAW],
                decimal=self.decimal,
            )

            # Test translation along y-axis
            dy = np.random.uniform(-5.0, 5.0)
            translated_se2_y = translate_se2_along_body_frame(pose_se2, Vector2D(0.0, dy))
            translated_se3_y = translate_euler_se3_along_y(pose_se3, dy)

            np.testing.assert_array_almost_equal(
                translated_se2_y.array[PoseSE2Index.XY],
                translated_se3_y.array[EulerPoseSE3Index.XY],
                decimal=self.decimal,
            )
            np.testing.assert_almost_equal(
                translated_se2_y.array[PoseSE2Index.YAW],
                translated_se3_y.array[EulerPoseSE3Index.YAW],
                decimal=self.decimal,
            )

            # Test translation along x- and y-axis
            dx = np.random.uniform(-5.0, 5.0)
            dy = np.random.uniform(-5.0, 5.0)
            translated_se2_xy = translate_se2_along_body_frame(pose_se2, Vector2D(dx, dy))
            translated_se3_xy = translate_euler_se3_along_body_frame(pose_se3, Vector3D(dx, dy, 0.0))
            np.testing.assert_array_almost_equal(
                translated_se2_xy.array[PoseSE2Index.XY],
                translated_se3_xy.array[EulerPoseSE3Index.XY],
                decimal=self.decimal,
            )
            np.testing.assert_almost_equal(
                translated_se2_xy.array[PoseSE2Index.YAW],
                translated_se3_xy.array[EulerPoseSE3Index.YAW],
                decimal=self.decimal,
            )

    def test_se2_se3_point_conversion_consistency(self) -> None:
        """Test that SE2 and SE3 point conversions are consistent for 2D points embedded in 3D"""
        for _ in range(self.num_consistency_tests):
            # Create equivalent SE2 and SE3 reference poses
            x = np.random.uniform(-10.0, 10.0)
            y = np.random.uniform(-10.0, 10.0)
            yaw = np.random.uniform(-np.pi, np.pi)

            reference_se2 = PoseSE2.from_array(np.array([x, y, yaw], dtype=np.float64))
            reference_se3 = EulerPoseSE3.from_array(np.array([x, y, 0.0, 0.0, 0.0, yaw], dtype=np.float64))

            # Generate 2D points and embed them in 3D with z=0
            num_points = np.random.randint(1, 8)
            points_2d = np.random.uniform(-20.0, 20.0, (num_points, len(Point2DIndex)))
            points_3d = np.column_stack([points_2d, np.zeros(num_points)])

            # Convert using SE2 functions
            relative_2d = convert_absolute_to_relative_points_2d_array(reference_se2, points_2d)
            absolute_2d_recovered = convert_relative_to_absolute_points_2d_array(reference_se2, relative_2d)

            # Convert using SE3 functions
            relative_3d = convert_absolute_to_relative_points_3d_array(reference_se3, points_3d)
            absolute_3d_recovered = convert_relative_to_absolute_points_3d_array(reference_se3, relative_3d)

            # Check that SE2 and SE3 conversions are consistent for the x,y components
            np.testing.assert_array_almost_equal(relative_2d, relative_3d[:, Point3DIndex.XY], decimal=self.decimal)
            np.testing.assert_array_almost_equal(
                absolute_2d_recovered, absolute_3d_recovered[:, Point3DIndex.XY], decimal=self.decimal
            )

            # Check that z-components remain zero
            np.testing.assert_array_almost_equal(
                relative_3d[:, Point3DIndex.Z], np.zeros(num_points), decimal=self.decimal
            )
            np.testing.assert_array_almost_equal(
                absolute_3d_recovered[:, Point3DIndex.Z], np.zeros(num_points), decimal=self.decimal
            )

    def test_se2_se3_pose_conversion_consistency(self) -> None:
        """Test that SE2 and SE3 pose conversions are consistent for 2D points embedded in 3D"""
        for _ in range(self.num_consistency_tests):
            # Create equivalent SE2 and SE3 reference poses
            x = np.random.uniform(-10.0, 10.0)
            y = np.random.uniform(-10.0, 10.0)
            yaw = np.random.uniform(-np.pi, np.pi)

            reference_se2 = PoseSE2.from_array(np.array([x, y, yaw], dtype=np.float64))
            reference_se3 = EulerPoseSE3.from_array(np.array([x, y, 0.0, 0.0, 0.0, yaw], dtype=np.float64))

            # Generate 2D poses and embed them in 3D with z=0 and zero roll/pitch
            num_poses = np.random.randint(1, 8)
            pose_2d = self._get_random_se2_array(num_poses)
            pose_3d = np.zeros((num_poses, len(EulerPoseSE3Index)), dtype=np.float64)
            pose_3d[:, EulerPoseSE3Index.XY] = pose_2d[:, PoseSE2Index.XY]
            pose_3d[:, EulerPoseSE3Index.YAW] = pose_2d[:, PoseSE2Index.YAW]

            # Convert using SE2 functions
            relative_se2 = convert_absolute_to_relative_se2_array(reference_se2, pose_2d)
            absolute_se2_recovered = convert_relative_to_absolute_se2_array(reference_se2, relative_se2)

            # Convert using SE3 functions
            relative_se3 = convert_absolute_to_relative_euler_se3_array(reference_se3, pose_3d)
            absolute_se3_recovered = convert_relative_to_absolute_euler_se3_array(reference_se3, relative_se3)

            # Check that SE2 and SE3 conversions are consistent for the x,y components
            np.testing.assert_array_almost_equal(
                relative_se2[:, PoseSE2Index.XY], relative_se3[:, EulerPoseSE3Index.XY], decimal=self.decimal
            )
            np.testing.assert_array_almost_equal(
                absolute_se2_recovered[:, PoseSE2Index.XY],
                absolute_se3_recovered[:, EulerPoseSE3Index.XY],
                decimal=self.decimal,
            )
            # Check that SE2 and SE3 conversions are consistent for the yaw component
            np.testing.assert_array_almost_equal(
                relative_se2[:, PoseSE2Index.YAW], relative_se3[:, EulerPoseSE3Index.YAW], decimal=self.decimal
            )
            np.testing.assert_array_almost_equal(
                absolute_se2_recovered[:, PoseSE2Index.YAW],
                absolute_se3_recovered[:, EulerPoseSE3Index.YAW],
                decimal=self.decimal,
            )

            # Check that z-components remain zero
            np.testing.assert_array_almost_equal(
                relative_se3[:, Point3DIndex.Z], np.zeros(num_poses), decimal=self.decimal
            )
            np.testing.assert_array_almost_equal(
                absolute_se3_recovered[:, Point3DIndex.Z], np.zeros(num_poses), decimal=self.decimal
            )

    def test_se2_array_translation_consistency(self) -> None:
        """Test that SE2 array translation is consistent with single pose translation"""
        for _ in range(self.num_consistency_tests):
            # Generate random poses
            num_poses = np.random.randint(self.min_random_poses, self.max_random_poses)
            poses_array = self._get_random_se2_array(num_poses)

            # Generate random translation
            dx = np.random.uniform(-5.0, 5.0)
            dy = np.random.uniform(-5.0, 5.0)
            translation = Vector2D(dx, dy)

            # Translate using array function
            result_array = translate_se2_array_along_body_frame(poses_array, translation)

            # Translate each pose individually
            result_individual = np.zeros_like(poses_array)
            for i in range(num_poses):
                pose = PoseSE2.from_array(poses_array[i])
                translated = translate_se2_along_body_frame(pose, translation)
                result_individual[i] = translated.array

            np.testing.assert_array_almost_equal(result_array, result_individual, decimal=self.decimal)

    def test_transform_empty_arrays(self) -> None:
        """Test that transform functions handle empty arrays correctly"""
        reference_se2 = PoseSE2.from_array(np.array([1.0, 2.0, np.pi / 4], dtype=np.float64))
        reference_se3 = EulerPoseSE3.from_array(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float64))

        # Test SE2 empty arrays
        empty_se2_poses = np.array([], dtype=np.float64).reshape(0, len(PoseSE2Index))
        empty_2d_points = np.array([], dtype=np.float64).reshape(0, len(Point2DIndex))

        result_se2_poses = convert_absolute_to_relative_se2_array(reference_se2, empty_se2_poses)
        result_2d_points = convert_absolute_to_relative_points_2d_array(reference_se2, empty_2d_points)

        assert result_se2_poses.shape == (0, len(PoseSE2Index))
        assert result_2d_points.shape == (0, len(Point2DIndex))

        # Test SE3 empty arrays
        empty_se3_poses = np.array([], dtype=np.float64).reshape(0, len(EulerPoseSE3Index))
        empty_3d_points = np.array([], dtype=np.float64).reshape(0, len(Point3DIndex))

        result_se3_poses = convert_absolute_to_relative_euler_se3_array(reference_se3, empty_se3_poses)
        result_3d_points = convert_absolute_to_relative_points_3d_array(reference_se3, empty_3d_points)

        assert result_se3_poses.shape == (0, len(EulerPoseSE3Index))
        assert result_3d_points.shape == (0, len(Point3DIndex))

    def test_transform_identity_operations(self) -> None:
        """Test that transforms with identity reference frames work correctly"""
        # Identity SE2 pose
        identity_se2 = PoseSE2.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        identity_se3 = EulerPoseSE3.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))

        for _ in range(self.num_consistency_tests):
            # Test SE2 identity transforms
            num_poses = np.random.randint(1, 10)
            se2_poses = self._get_random_se2_array(num_poses)
            se2_points = se2_poses[:, PoseSE2Index.XY]

            relative_se2_poses = convert_absolute_to_relative_se2_array(identity_se2, se2_poses)
            relative_se2_points = convert_absolute_to_relative_points_2d_array(identity_se2, se2_points)

            np.testing.assert_array_almost_equal(se2_poses, relative_se2_poses, decimal=self.decimal)
            np.testing.assert_array_almost_equal(se2_points, relative_se2_points, decimal=self.decimal)

            # Test SE3 identity transforms
            se3_poses = self._get_random_se3_array(num_poses)
            se3_points = se3_poses[:, EulerPoseSE3Index.XYZ]

            relative_se3_poses = convert_absolute_to_relative_euler_se3_array(identity_se3, se3_poses)
            relative_se3_points = convert_absolute_to_relative_points_3d_array(identity_se3, se3_points)

            np.testing.assert_array_almost_equal(
                se3_poses[..., EulerPoseSE3Index.EULER_ANGLES],
                relative_se3_poses[..., EulerPoseSE3Index.EULER_ANGLES],
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(se3_points, relative_se3_points, decimal=self.decimal)

    def test_transform_large_rotations(self) -> None:
        """Test transforms with large rotation angles beyond [-π, π]"""
        for _ in range(self.num_consistency_tests):
            # Create poses with large rotation angles
            large_yaw_se2 = np.random.uniform(-4 * np.pi, 4 * np.pi)
            large_euler_se3 = np.random.uniform(-4 * np.pi, 4 * np.pi, 3)

            reference_se2 = PoseSE2.from_array(np.array([0.0, 0.0, large_yaw_se2], dtype=np.float64))
            reference_se3 = EulerPoseSE3.from_array(
                np.array([0.0, 0.0, 0.0, large_euler_se3[0], large_euler_se3[1], large_euler_se3[2]], dtype=np.float64)
            )

            # Generate test poses/points
            test_se2_poses = self._get_random_se2_array(5)
            test_se3_poses = self._get_random_se3_array(5)
            test_2d_points = test_se2_poses[:, PoseSE2Index.XY]
            test_3d_points = test_se3_poses[:, EulerPoseSE3Index.XYZ]

            # Test round-trip conversions should still work
            relative_se2 = convert_absolute_to_relative_se2_array(reference_se2, test_se2_poses)
            recovered_se2 = convert_relative_to_absolute_se2_array(reference_se2, relative_se2)

            relative_se3 = convert_absolute_to_relative_euler_se3_array(reference_se3, test_se3_poses)
            recovered_se3 = convert_relative_to_absolute_euler_se3_array(reference_se3, relative_se3)

            relative_2d_points = convert_absolute_to_relative_points_2d_array(reference_se2, test_2d_points)
            recovered_2d_points = convert_relative_to_absolute_points_2d_array(reference_se2, relative_2d_points)

            relative_3d_points = convert_absolute_to_relative_points_3d_array(reference_se3, test_3d_points)
            recovered_3d_points = convert_relative_to_absolute_points_3d_array(reference_se3, relative_3d_points)

            # Check consistency (allowing for angle wrapping)
            np.testing.assert_array_almost_equal(
                test_se2_poses[:, PoseSE2Index.XY],
                recovered_se2[:, PoseSE2Index.XY],
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(
                test_se3_poses[:, EulerPoseSE3Index.XYZ],
                recovered_se3[:, EulerPoseSE3Index.XYZ],
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(test_2d_points, recovered_2d_points, decimal=self.decimal)
            np.testing.assert_array_almost_equal(test_3d_points, recovered_3d_points, decimal=self.decimal)
