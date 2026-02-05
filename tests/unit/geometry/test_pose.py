import numpy as np
import pytest

from py123d.geometry import Point2D, PoseSE2, PoseSE3
from py123d.geometry.geometry_index import PoseSE2Index
from py123d.geometry.pose import EulerPoseSE3


class TestPoseSE2:
    def test_init(self):
        """Test basic initialization with explicit x, y, yaw values."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.yaw == 0.5

    def test_from_array(self):
        """Test creation from numpy array."""
        array = np.array([1.0, 2.0, 0.5])
        pose = PoseSE2.from_array(array)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.yaw == 0.5

    def test_from_array_copy(self):
        """Test that copy=True creates independent pose from array."""
        array = np.array([1.0, 2.0, 0.5])
        pose = PoseSE2.from_array(array, copy=True)
        array[0] = 99.0
        assert pose.x == 1.0

    def test_from_array_no_copy(self):
        """Test that copy=False links pose to original array."""
        array = np.array([1.0, 2.0, 0.5])
        pose = PoseSE2.from_array(array, copy=False)
        array[0] = 99.0
        assert pose.x == 99.0

    def test_from_identity(self):
        """Test creation of identity pose."""
        pose = PoseSE2.identity()
        assert pose.x == 0.0
        assert pose.y == 0.0
        assert pose.yaw == 0.0
        transformation_matrix = pose.transformation_matrix
        np.testing.assert_allclose(transformation_matrix, np.eye(3), atol=1e-10)

    def test_properties(self):
        """Test access to individual pose component properties."""
        pose = PoseSE2(x=3.0, y=4.0, yaw=np.pi / 4)
        assert pose.x == 3.0
        assert pose.y == 4.0
        assert pytest.approx(pose.yaw) == np.pi / 4

    def test_array_property(self):
        """Test that the array property returns correct numpy array."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        array = pose.array
        assert array.shape == (3,)
        assert array[PoseSE2Index.X] == 1.0
        assert array[PoseSE2Index.Y] == 2.0
        assert array[PoseSE2Index.YAW] == 0.5

    def test_point_2d(self):
        """Test extraction of 2D position as Point2D."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        point = pose.point_2d
        assert isinstance(point, Point2D)
        assert point.x == 1.0
        assert point.y == 2.0

    def test_rotation_matrix(self):
        """Test extraction of 2x2 rotation matrix."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.0)
        rot_mat = pose.rotation_matrix
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(rot_mat, expected)

    def test_rotation_matrix_pi_half(self):
        """Test extraction of 2x2 rotation matrix for 90 degree rotation."""
        pose = PoseSE2(x=0.0, y=0.0, yaw=np.pi / 2)
        rot_mat = pose.rotation_matrix
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_allclose(rot_mat, expected, atol=1e-10)

    def test_transformation_matrix(self):
        """Test extraction of 3x3 transformation matrix."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.0)
        trans_mat = pose.transformation_matrix
        assert trans_mat.shape == (3, 3)
        expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(trans_mat, expected)

    def test_shapely_point(self):
        """Test extraction of Shapely Point representation."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        shapely_point = pose.shapely_point
        assert shapely_point.x == 1.0
        assert shapely_point.y == 2.0

    def test_pose_se2_property(self):
        """Test that pose_se2 property returns self."""
        pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        assert pose.pose_se2 is pose

    def test_equality(self):
        """Test equality comparison of PoseSE2 instances."""
        pose1 = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        pose2 = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        assert pose1 == pose2

    def test_inequality(self):
        """Test inequality comparison of PoseSE2 instances."""
        pose1 = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        pose2 = PoseSE2(x=1.0, y=2.0, yaw=0.6)
        assert pose1 != pose2


class TestPoseSE3:
    def test_init(self):
        """Test basic initialization with explicit x, y, z, and quaternion values."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_from_array(self):
        """Test creation from numpy array."""
        array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        pose = PoseSE3.from_array(array)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_from_array_copy(self):
        """Test that copy=True creates independent pose from array."""
        array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        pose = PoseSE3.from_array(array, copy=True)
        array[0] = 99.0
        assert pose.x == 1.0

    def test_from_array_no_copy(self):
        """Test that copy=False links pose to original array."""
        array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        pose = PoseSE3.from_array(array, copy=False)
        array[0] = 99.0
        assert pose.x == 99.0

    def test_from_transformation_matrix(self):
        """Test creation from 4x4 transformation matrix."""
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = [1.0, 2.0, 3.0]
        pose = PoseSE3.from_transformation_matrix(trans_mat)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_from_identity(self):
        """Test creation of identity pose."""
        pose = PoseSE3.identity()
        np.testing.assert_allclose(pose.array, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), atol=1e-10)
        np.testing.assert_allclose(pose.transformation_matrix, np.eye(4), atol=1e-10)
        assert pose.yaw == 0.0
        assert pose.pitch == 0.0
        assert pose.roll == 0.0

    def test_properties(self):
        """Test access to individual pose component properties."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.qx == 0.0
        assert pose.qy == 0.0
        assert pose.qz == 0.0

    def test_array_property(self):
        """Test that the array property returns the correct numpy array representation."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        array = pose.array
        assert array.shape == (7,)
        np.testing.assert_allclose(array, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    def test_pose_se2(self):
        """Test extraction of 2D pose from 3D pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        pose_2d = pose.pose_se2
        assert isinstance(pose_2d, PoseSE2)
        assert pose_2d.x == 1.0
        assert pose_2d.y == 2.0
        assert pytest.approx(pose_2d.yaw) == 0.0

    def test_point_3d(self):
        """Test extraction of 3D point from 3D pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        point = pose.point_3d
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.z == 3.0

    def test_point_2d(self):
        """Test extraction of 2D point from 3D pose."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        point = pose.point_2d
        assert isinstance(point, Point2D)
        assert point.x == 1.0
        assert point.y == 2.0

    def test_shapely_point(self):
        """Test extraction of Shapely Point representation."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        shapely_point = pose.shapely_point
        assert shapely_point.x == 1.0
        assert shapely_point.y == 2.0
        assert shapely_point.z == 3.0

    def test_rotation_matrix(self):
        """Test extraction of 3x3 rotation matrix."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        rot_mat = pose.rotation_matrix
        expected = np.eye(3)
        np.testing.assert_allclose(rot_mat, expected)

    def test_transformation_matrix(self):
        """Test extraction of 4x4 transformation matrix."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        trans_mat = pose.transformation_matrix
        assert trans_mat.shape == (4, 4)
        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(trans_mat, expected)

    def test_transformation_matrix_roundtrip(self):
        """Test round-trip conversion between pose and transformation matrix."""
        pose1 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        trans_mat = pose1.transformation_matrix
        pose2 = PoseSE3.from_transformation_matrix(trans_mat)
        assert pose1 == pose2

    def test_euler_angles(self):
        """Test extraction of Euler angles from quaternion."""
        pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pytest.approx(pose.roll) == 0.0
        assert pytest.approx(pose.pitch) == 0.0
        assert pytest.approx(pose.yaw) == 0.0

    def test_equality(self):
        """Test equality comparison of PoseSE3 instances."""
        pose1 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        pose2 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        assert pose1 == pose2

    def test_inequality(self):
        """Test inequality comparison of PoseSE3 instances."""
        pose1 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        pose2 = PoseSE3(x=1.0, y=2.0, z=3.0, qw=0.9, qx=0.1, qy=0.0, qz=0.0)
        assert pose1 != pose2


class TestEulerPoseSE3:
    def test_init(self):
        """Test initialization of EulerPoseSE3 with position and orientation."""

        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.roll == 0.1
        assert pose.pitch == 0.2
        assert pose.yaw == 0.3

    def test_from_array(self):
        """Test creation of EulerPoseSE3 from numpy array."""

        array = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        pose = EulerPoseSE3.from_array(array)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.roll == 0.1
        assert pose.pitch == 0.2
        assert pose.yaw == 0.3

    def test_from_array_copy(self):
        """Test that copy=True creates independent pose from array."""

        array = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        pose = EulerPoseSE3.from_array(array, copy=True)
        array[0] = 99.0
        assert pose.x == 1.0

    def test_from_array_no_copy(self):
        """Test that copy=False links pose to original array."""

        array = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        pose = EulerPoseSE3.from_array(array, copy=False)
        array[0] = 99.0
        assert pose.x == 99.0

    def test_from_transformation_matrix(self):
        """Test creation of EulerPoseSE3 from 4x4 transformation matrix."""
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = [1.0, 2.0, 3.0]
        pose = EulerPoseSE3.from_transformation_matrix(trans_mat)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pytest.approx(pose.roll) == 0.0
        assert pytest.approx(pose.pitch) == 0.0
        assert pytest.approx(pose.yaw) == 0.0

    def test_properties(self):
        """Test access to individual pose component properties."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.roll == 0.1
        assert pose.pitch == 0.2
        assert pose.yaw == 0.3

    def test_array_property(self):
        """Test that the array property returns the correct numpy array representation."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        array = pose.array
        assert array.shape == (6,)
        np.testing.assert_allclose(array, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    def test_pose_se2(self):
        """Test extraction of 2D pose from 3D Euler pose."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        pose_2d = pose.pose_se2
        assert isinstance(pose_2d, PoseSE2)
        assert pose_2d.x == 1.0
        assert pose_2d.y == 2.0
        assert pose_2d.yaw == 0.3

    def test_point_3d(self):
        """Test extraction of 3D point from 3D Euler pose."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        point = pose.point_3d
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.z == 3.0

    def test_point_2d(self):
        """Test extraction of 2D point from 3D Euler pose."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        point = pose.point_2d
        assert isinstance(point, Point2D)
        assert point.x == 1.0
        assert point.y == 2.0

    def test_shapely_point(self):
        """Test extraction of Shapely Point representation."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        shapely_point = pose.shapely_point
        assert shapely_point.x == 1.0
        assert shapely_point.y == 2.0
        assert shapely_point.z == 3.0

    def test_rotation_matrix(self):
        """Test the rotation matrix property of EulerPoseSE3."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=0.0)
        rot_mat = pose.rotation_matrix
        expected = np.eye(3)
        np.testing.assert_allclose(rot_mat, expected)

    def test_transformation_matrix(self):
        """Test the transformation matrix property of EulerPoseSE3."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=0.0)
        trans_mat = pose.transformation_matrix
        assert trans_mat.shape == (4, 4)
        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(trans_mat, expected)

    def test_transformation_matrix_roundtrip(self):
        """Test round-trip conversion between EulerPoseSE3 and transformation matrix."""
        pose1 = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=0.0)
        trans_mat = pose1.transformation_matrix
        pose2 = EulerPoseSE3.from_transformation_matrix(trans_mat)
        np.testing.assert_allclose(pose1.array, pose2.array)

    def test_euler_angles(self):
        """Test the euler_angles property of EulerPoseSE3."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        euler = pose.euler_angles
        assert pytest.approx(euler.roll) == 0.1
        assert pytest.approx(euler.pitch) == 0.2
        assert pytest.approx(euler.yaw) == 0.3

    def test_pose_se3_conversion(self):
        """Test conversion from EulerPoseSE3 to PoseSE3."""
        euler_pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=0.0)
        quat_pose = euler_pose.pose_se3
        assert isinstance(quat_pose, PoseSE3)
        assert quat_pose.x == 1.0
        assert quat_pose.y == 2.0
        assert quat_pose.z == 3.0
        assert pytest.approx(quat_pose.qw) == 1.0
        assert pytest.approx(quat_pose.qx) == 0.0
        assert pytest.approx(quat_pose.qy) == 0.0
        assert pytest.approx(quat_pose.qz) == 0.0

    def test_quaternion(self):
        """Test the quaternion property of EulerPoseSE3."""
        pose = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=0.0)
        quat = pose.quaternion
        assert pytest.approx(quat.qw) == 1.0
        assert pytest.approx(quat.qx) == 0.0
        assert pytest.approx(quat.qy) == 0.0
        assert pytest.approx(quat.qz) == 0.0

    def test_equality(self):
        """Test equality comparison of EulerPoseSE3 instances."""

        pose1 = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        pose2 = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        assert pose1 == pose2

    def test_inequality(self):
        """Test inequality comparison of EulerPoseSE3 instances."""

        pose1 = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        pose2 = EulerPoseSE3(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.4)
        assert pose1 != pose2
