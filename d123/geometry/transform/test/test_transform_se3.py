import unittest

import numpy as np
import numpy.typing as npt

from d123.geometry.geometry_index import StateSE3Index, EulerStateSE3Index
from d123.geometry.point import Point3D
from d123.geometry.rotation import Quaternion
from d123.geometry.se import EulerStateSE3, StateSE3
from d123.geometry.transform.transform_se3 import (
    convert_absolute_to_relative_points_3d_array,
    convert_absolute_to_relative_se3_array,
    convert_relative_to_absolute_points_3d_array,
    convert_relative_to_absolute_se3_array,
    translate_se3_along_x,
    translate_se3_along_y,
    translate_se3_along_z,
    translate_se3_along_body_frame,
)
import d123.geometry.transform.transform_euler_se3 as euler_transform_se3
from d123.geometry.utils.rotation_utils import (
    get_rotation_matrices_from_euler_array,
    get_rotation_matrices_from_quaternion_array,
)


class TestTransformSE3(unittest.TestCase):

    def setUp(self):
        euler_se3_a = EulerStateSE3(
            x=1.0,
            y=2.0,
            z=3.0,
            roll=np.deg2rad(90),
            pitch=0.0,
            yaw=0.0,
        )
        euler_se3_b = EulerStateSE3(
            x=1.0,
            y=-2.0,
            z=3.0,
            roll=0.0,
            pitch=np.deg2rad(90),
            yaw=0.0,
        )
        euler_se3_c = EulerStateSE3(
            x=-1.0,
            y=2.0,
            z=-3.0,
            roll=0.0,
            pitch=0.0,
            yaw=np.deg2rad(90),
        )

        quat_se3_a: StateSE3 = euler_se3_a.quaternion_se3
        quat_se3_b: StateSE3 = euler_se3_b.quaternion_se3
        quat_se3_c: StateSE3 = euler_se3_c.quaternion_se3

        self.euler_se3 = [euler_se3_a, euler_se3_b, euler_se3_c]
        self.quat_se3 = [quat_se3_a, quat_se3_b, quat_se3_c]

        self.max_pose_xyz = 100.0

    def _get_random_euler_se3_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate a random SE3 poses"""
        random_se3_array = np.zeros((size, len(EulerStateSE3Index)), dtype=np.float64)
        random_se3_array[:, EulerStateSE3Index.XYZ] = np.random.uniform(
            -self.max_pose_xyz, self.max_pose_xyz, (size, 3)
        )
        random_se3_array[:, EulerStateSE3Index.YAW] = np.random.uniform(-np.pi, np.pi, size)
        random_se3_array[:, EulerStateSE3Index.PITCH] = np.random.uniform(-np.pi / 2, np.pi / 2, size)
        random_se3_array[:, EulerStateSE3Index.ROLL] = np.random.uniform(-np.pi, np.pi, size)

        return random_se3_array

    def _convert_euler_se3_array_to_quat_se3_array(
        self, euler_se3_array: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Convert an array of SE3 poses from Euler angles to Quaternion representation"""
        quat_se3_array = np.zeros((euler_se3_array.shape[0], len(StateSE3Index)), dtype=np.float64)
        quat_se3_array[:, StateSE3Index.XYZ] = euler_se3_array[:, EulerStateSE3Index.XYZ]
        rotation_matrices = get_rotation_matrices_from_euler_array(euler_se3_array[:, EulerStateSE3Index.EULER_ANGLES])
        for idx, rotation_matrix in enumerate(rotation_matrices):
            quat = Quaternion.from_rotation_matrix(rotation_matrix)
            quat_se3_array[idx, StateSE3Index.QUATERNION] = quat.array
        return quat_se3_array

    def test_sanity(self):
        for quat_se3, euler_se3 in zip(self.quat_se3, self.euler_se3):
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
                random_euler_se3_array[:, EulerStateSE3Index.XYZ],
                random_quat_se3_array[:, StateSE3Index.XYZ],
                atol=1e-6,
            )
            quat_rotation_matrices = get_rotation_matrices_from_quaternion_array(
                random_quat_se3_array[:, StateSE3Index.QUATERNION]
            )
            euler_rotation_matrices = get_rotation_matrices_from_euler_array(
                random_euler_se3_array[:, EulerStateSE3Index.EULER_ANGLES]
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
                rel_se3_euler[..., EulerStateSE3Index.XYZ], rel_se3_quat[..., StateSE3Index.XYZ], atol=1e-6
            )
            # We compare rotation matrices to avoid issues with quaternion sign ambiguity
            quat_rotation_matrices = get_rotation_matrices_from_quaternion_array(
                rel_se3_quat[..., StateSE3Index.QUATERNION]
            )
            euler_rotation_matrices = get_rotation_matrices_from_euler_array(
                rel_se3_euler[..., EulerStateSE3Index.EULER_ANGLES]
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
                abs_se3_euler[..., EulerStateSE3Index.XYZ], abs_se3_quat[..., StateSE3Index.XYZ], atol=1e-6
            )
            # We compare rotation matrices to avoid issues with quaternion sign ambiguity
            quat_rotation_matrices = get_rotation_matrices_from_quaternion_array(
                abs_se3_quat[..., StateSE3Index.QUATERNION]
            )
            euler_rotation_matrices = get_rotation_matrices_from_euler_array(
                abs_se3_euler[..., EulerStateSE3Index.EULER_ANGLES]
            )
            np.testing.assert_allclose(quat_rotation_matrices, euler_rotation_matrices, atol=1e-6)

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


if __name__ == "__main__":
    unittest.main()
