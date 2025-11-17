import numpy as np
import numpy.typing as npt

import py123d.geometry.transform.transform_euler_se3 as euler_transform_se3
from py123d.geometry import EulerPoseSE3, EulerPoseSE3Index, Point3D, PoseSE3, PoseSE3Index
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
from py123d.geometry.utils.rotation_utils import (
    get_quaternion_array_from_euler_array,
    get_rotation_matrices_from_euler_array,
    get_rotation_matrices_from_quaternion_array,
)


class TestTransformSE3:
    def setup_method(self):
        euler_se3_a = EulerPoseSE3(
            x=1.0,
            y=2.0,
            z=3.0,
            roll=np.deg2rad(90),
            pitch=0.0,
            yaw=0.0,
        )
        euler_se3_b = EulerPoseSE3(
            x=1.0,
            y=-2.0,
            z=3.0,
            roll=0.0,
            pitch=np.deg2rad(90),
            yaw=0.0,
        )
        euler_se3_c = EulerPoseSE3(
            x=-1.0,
            y=2.0,
            z=-3.0,
            roll=0.0,
            pitch=0.0,
            yaw=np.deg2rad(90),
        )

        quat_se3_a: PoseSE3 = euler_se3_a.pose_se3
        quat_se3_b: PoseSE3 = euler_se3_b.pose_se3
        quat_se3_c: PoseSE3 = euler_se3_c.pose_se3

        self.euler_se3 = [euler_se3_a, euler_se3_b, euler_se3_c]
        self.quat_se3 = [quat_se3_a, quat_se3_b, quat_se3_c]

        self.max_pose_xyz = 100.0

    def _get_random_euler_se3_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate a random SE3 poses"""
        random_se3_array = np.zeros((size, len(EulerPoseSE3Index)), dtype=np.float64)
        random_se3_array[:, EulerPoseSE3Index.XYZ] = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz, (size, 3))
        random_se3_array[:, EulerPoseSE3Index.YAW] = np.random.uniform(-np.pi, np.pi, size)
        random_se3_array[:, EulerPoseSE3Index.PITCH] = np.random.uniform(-np.pi / 2, np.pi / 2, size)
        random_se3_array[:, EulerPoseSE3Index.ROLL] = np.random.uniform(-np.pi, np.pi, size)

        return random_se3_array

    def _convert_euler_se3_array_to_quat_se3_array(
        self, euler_se3_array: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Convert an array of SE3 poses from Euler angles to Quaternion representation"""
        quat_se3_array = np.zeros((euler_se3_array.shape[0], len(PoseSE3Index)), dtype=np.float64)
        quat_se3_array[:, PoseSE3Index.XYZ] = euler_se3_array[:, EulerPoseSE3Index.XYZ]
        quat_se3_array[:, PoseSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(
            euler_se3_array[:, EulerPoseSE3Index.EULER_ANGLES]
        )
        return quat_se3_array

    def _get_random_quat_se3_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate a random SE3 poses in Quaternion representation"""
        random_euler_se3_array = self._get_random_euler_se3_array(size)
        random_quat_se3_array = self._convert_euler_se3_array_to_quat_se3_array(random_euler_se3_array)
        return random_quat_se3_array

    def test_sanity(self):
        for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
            np.testing.assert_allclose(
                quat_se3.point_3d.array,
                euler_se3.point_3d.array,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                quat_se3.rotation_matrix,
                euler_se3.rotation_matrix,
                atol=1e-6,
            )

    def test_random_sanity(self):
        for _ in range(10):
            random_euler_se3_array = self._get_random_euler_se3_array(np.random.randint(1, 10))
            random_quat_se3_array = self._convert_euler_se3_array_to_quat_se3_array(random_euler_se3_array)

            np.testing.assert_allclose(
                random_euler_se3_array[:, EulerPoseSE3Index.XYZ],
                random_quat_se3_array[:, PoseSE3Index.XYZ],
                atol=1e-6,
            )
            quat_rotation_matrices = get_rotation_matrices_from_quaternion_array(
                random_quat_se3_array[:, PoseSE3Index.QUATERNION]
            )
            euler_rotation_matrices = get_rotation_matrices_from_euler_array(
                random_euler_se3_array[:, EulerPoseSE3Index.EULER_ANGLES]
            )
            np.testing.assert_allclose(euler_rotation_matrices, quat_rotation_matrices, atol=1e-6)

    def test_convert_absolute_to_relative_points_3d_array(self):
        random_points_3d = np.random.rand(10, 3)
        for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
            rel_points_quat = convert_absolute_to_relative_points_3d_array(quat_se3, random_points_3d)
            rel_points_euler = euler_transform_se3.convert_absolute_to_relative_points_3d_array(
                euler_se3, random_points_3d
            )
            np.testing.assert_allclose(rel_points_quat, rel_points_euler, atol=1e-6)

    def test_convert_absolute_to_relative_se3_array(self):
        for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
            random_euler_se3_array = self._get_random_euler_se3_array(np.random.randint(1, 10))
            random_quat_se3_array = self._convert_euler_se3_array_to_quat_se3_array(random_euler_se3_array)

            rel_se3_quat = convert_absolute_to_relative_se3_array(quat_se3, random_quat_se3_array)
            rel_se3_euler = euler_transform_se3.convert_absolute_to_relative_euler_se3_array(
                euler_se3, random_euler_se3_array
            )
            np.testing.assert_allclose(
                rel_se3_euler[..., EulerPoseSE3Index.XYZ], rel_se3_quat[..., PoseSE3Index.XYZ], atol=1e-6
            )
            # We compare rotation matrices to avoid issues with quaternion sign ambiguity
            quat_rotation_matrices = get_rotation_matrices_from_quaternion_array(
                rel_se3_quat[..., PoseSE3Index.QUATERNION]
            )
            euler_rotation_matrices = get_rotation_matrices_from_euler_array(
                rel_se3_euler[..., EulerPoseSE3Index.EULER_ANGLES]
            )
            np.testing.assert_allclose(quat_rotation_matrices, euler_rotation_matrices, atol=1e-6)

    def test_convert_relative_to_absolute_points_3d_array(self):
        random_points_3d = np.random.rand(10, 3)
        for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
            rel_points_quat = convert_relative_to_absolute_points_3d_array(quat_se3, random_points_3d)
            rel_points_euler = euler_transform_se3.convert_relative_to_absolute_points_3d_array(
                euler_se3, random_points_3d
            )
            np.testing.assert_allclose(rel_points_quat, rel_points_euler, atol=1e-6)

    def test_convert_relative_to_absolute_se3_array(self):
        for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
            random_euler_se3_array = self._get_random_euler_se3_array(np.random.randint(1, 10))
            random_quat_se3_array = self._convert_euler_se3_array_to_quat_se3_array(random_euler_se3_array)

            abs_se3_quat = convert_relative_to_absolute_se3_array(quat_se3, random_quat_se3_array)
            abs_se3_euler = euler_transform_se3.convert_relative_to_absolute_euler_se3_array(
                euler_se3, random_euler_se3_array
            )
            np.testing.assert_allclose(
                abs_se3_euler[..., EulerPoseSE3Index.XYZ], abs_se3_quat[..., PoseSE3Index.XYZ], atol=1e-6
            )

            # We compare rotation matrices to avoid issues with quaternion sign ambiguity
            quat_rotation_matrices = get_rotation_matrices_from_quaternion_array(
                abs_se3_quat[..., PoseSE3Index.QUATERNION]
            )
            euler_rotation_matrices = get_rotation_matrices_from_euler_array(
                abs_se3_euler[..., EulerPoseSE3Index.EULER_ANGLES]
            )
            np.testing.assert_allclose(quat_rotation_matrices, euler_rotation_matrices, atol=1e-6)
            # convert_points_3d_array_between_origins(quat_se3, random_quat_se3_array)

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
            for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
                translated_quat = translate_se3_along_x(quat_se3, distance)
                translated_euler = euler_transform_se3.translate_euler_se3_along_x(euler_se3, distance)
                np.testing.assert_allclose(translated_quat.point_3d.array, translated_euler.point_3d.array, atol=1e-6)
                np.testing.assert_allclose(translated_quat.rotation_matrix, translated_euler.rotation_matrix, atol=1e-6)
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)

    def test_translate_se3_along_y(self):
        for _ in range(10):
            distance = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz)
            for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
                translated_quat = translate_se3_along_y(quat_se3, distance)
                translated_euler = euler_transform_se3.translate_euler_se3_along_y(euler_se3, distance)
                np.testing.assert_allclose(translated_quat.point_3d.array, translated_euler.point_3d.array, atol=1e-6)
                np.testing.assert_allclose(translated_quat.rotation_matrix, translated_euler.rotation_matrix, atol=1e-6)
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)

    def test_translate_se3_along_z(self):
        for _ in range(10):
            distance = np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz)
            for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
                translated_quat = translate_se3_along_z(quat_se3, distance)
                translated_euler = euler_transform_se3.translate_euler_se3_along_z(euler_se3, distance)
                np.testing.assert_allclose(translated_quat.point_3d.array, translated_euler.point_3d.array, atol=1e-6)
                np.testing.assert_allclose(translated_quat.rotation_matrix, translated_euler.rotation_matrix, atol=1e-6)
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)

    def test_translate_se3_along_body_frame(self):
        for _ in range(10):
            vector_3d = Point3D(
                x=np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz),
                y=np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz),
                z=np.random.uniform(-self.max_pose_xyz, self.max_pose_xyz),
            )
            for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
                translated_quat = translate_se3_along_body_frame(quat_se3, vector_3d)
                translated_euler = euler_transform_se3.translate_euler_se3_along_body_frame(euler_se3, vector_3d)
                np.testing.assert_allclose(translated_quat.point_3d.array, translated_euler.point_3d.array, atol=1e-6)
                np.testing.assert_allclose(translated_quat.rotation_matrix, translated_euler.rotation_matrix, atol=1e-6)
                np.testing.assert_allclose(quat_se3.quaternion.array, translated_quat.quaternion.array, atol=1e-6)
