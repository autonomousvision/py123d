import numpy as np
import numpy.typing as npt

from py123d.geometry import EulerAngles, Point3D, PoseSE3, PoseSE3Index
from py123d.geometry.transform.transform_se3 import (
    convert_absolute_to_relative_points_3d_array,
    convert_absolute_to_relative_se3_array,
    convert_points_3d_array_between_origins,
    convert_relative_to_absolute_points_3d_array,
    convert_relative_to_absolute_se3_array,
    convert_se3_array_between_origins,
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

    def test_convert_absolute_to_relative_points_3d_array(self):
        random_points_3d = np.random.rand(10, 3)
        for quat_se3 in self.quat_se3:
            rel_points = convert_absolute_to_relative_points_3d_array(quat_se3, random_points_3d)
            # Round-trip check
            abs_points = convert_relative_to_absolute_points_3d_array(quat_se3, rel_points)
            np.testing.assert_allclose(abs_points, random_points_3d, atol=1e-6)

    def test_convert_absolute_to_relative_se3_array(self):
        for quat_se3 in self.quat_se3:
            random_quat_se3_array = self._get_random_quat_se3_array(np.random.randint(1, 10))

            rel_se3_quat = convert_absolute_to_relative_se3_array(quat_se3, random_quat_se3_array)
            # Round-trip check
            abs_se3_quat = convert_relative_to_absolute_se3_array(quat_se3, rel_se3_quat)
            np.testing.assert_allclose(
                abs_se3_quat[..., PoseSE3Index.XYZ], random_quat_se3_array[..., PoseSE3Index.XYZ], atol=1e-6
            )

    def test_convert_relative_to_absolute_points_3d_array(self):
        random_points_3d = np.random.rand(10, 3)
        for quat_se3 in self.quat_se3:
            abs_points = convert_relative_to_absolute_points_3d_array(quat_se3, random_points_3d)
            # Round-trip check
            rel_points = convert_absolute_to_relative_points_3d_array(quat_se3, abs_points)
            np.testing.assert_allclose(rel_points, random_points_3d, atol=1e-6)

    def test_convert_relative_to_absolute_se3_array(self):
        for quat_se3 in self.quat_se3:
            random_quat_se3_array = self._get_random_quat_se3_array(np.random.randint(1, 10))

            abs_se3_quat = convert_relative_to_absolute_se3_array(quat_se3, random_quat_se3_array)
            # Round-trip check
            rel_se3_quat = convert_absolute_to_relative_se3_array(quat_se3, abs_se3_quat)
            np.testing.assert_allclose(
                rel_se3_quat[..., PoseSE3Index.XYZ], random_quat_se3_array[..., PoseSE3Index.XYZ], atol=1e-6
            )

    def test_convert_se3_array_between_origins(self):
        for _ in range(10):
            random_quat_se3_array = self._get_random_quat_se3_array(np.random.randint(1, 10))

            from_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            to_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            identity_se3_array = np.zeros(len(PoseSE3Index), dtype=np.float64)
            identity_se3_array[PoseSE3Index.QW] = 1.0
            identity_se3 = PoseSE3.from_array(identity_se3_array)

            # Check if consistent with absolute-relative-absolute conversion
            converted_se3_quat = convert_se3_array_between_origins(from_se3, to_se3, random_quat_se3_array)

            abs_from_se3_quat = convert_relative_to_absolute_se3_array(from_se3, random_quat_se3_array)
            rel_to_se3_quat = convert_absolute_to_relative_se3_array(to_se3, abs_from_se3_quat)

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
            absolute_se3_quat = convert_se3_array_between_origins(from_se3, identity_se3, random_quat_se3_array)
            np.testing.assert_allclose(
                absolute_se3_quat[..., PoseSE3Index.XYZ],
                abs_from_se3_quat[..., PoseSE3Index.XYZ],
                atol=1e-6,
            )

    def test_convert_points_3d_array_between_origins(self):
        random_points_3d = np.random.rand(10, 3)
        for _ in range(10):
            from_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            to_se3 = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            identity_se3_array = np.zeros(len(PoseSE3Index), dtype=np.float64)
            identity_se3_array[PoseSE3Index.QW] = 1.0
            identity_se3 = PoseSE3.from_array(identity_se3_array)

            # Check if consistent with absolute-relative-absolute conversion
            converted_points_quat = convert_points_3d_array_between_origins(from_se3, to_se3, random_points_3d)
            abs_from_se3_quat = convert_relative_to_absolute_points_3d_array(from_se3, random_points_3d)
            rel_to_se3_quat = convert_absolute_to_relative_points_3d_array(to_se3, abs_from_se3_quat)
            np.testing.assert_allclose(converted_points_quat, rel_to_se3_quat, atol=1e-6)

            # Check if consistent with se3 array conversion
            random_se3_poses = np.zeros((random_points_3d.shape[0], len(PoseSE3Index)), dtype=np.float64)
            random_se3_poses[:, PoseSE3Index.XYZ] = random_points_3d
            random_se3_poses[:, PoseSE3Index.QUATERNION] = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
            converted_se3_quat_poses = convert_se3_array_between_origins(from_se3, to_se3, random_se3_poses)
            np.testing.assert_allclose(
                converted_se3_quat_poses[:, PoseSE3Index.XYZ],
                converted_points_quat,
                atol=1e-6,
            )

            # Check if consistent with absolute conversion to identity origin
            absolute_se3_quat = convert_points_3d_array_between_origins(from_se3, identity_se3, random_points_3d)
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
