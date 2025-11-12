import unittest

import numpy as np

from py123d.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex
from py123d.geometry.rotation import EulerAngles, Quaternion


class TestEulerAngles(unittest.TestCase):
    """Unit tests for EulerAngles class."""

    def setUp(self):
        """Set up test fixtures."""
        self.roll = 0.1
        self.pitch = 0.2
        self.yaw = 0.3
        self.euler_angles = EulerAngles(self.roll, self.pitch, self.yaw)
        self.test_array = np.zeros([3], dtype=np.float64)
        self.test_array[EulerAnglesIndex.ROLL] = self.roll
        self.test_array[EulerAnglesIndex.PITCH] = self.pitch
        self.test_array[EulerAnglesIndex.YAW] = self.yaw

    def test_init(self):
        """Test EulerAngles initialization."""
        euler = EulerAngles(roll=0.1, pitch=0.2, yaw=0.3)
        self.assertEqual(euler.roll, 0.1)
        self.assertEqual(euler.pitch, 0.2)
        self.assertEqual(euler.yaw, 0.3)

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        euler = EulerAngles.from_array(self.test_array)
        self.assertIsInstance(euler, EulerAngles)
        self.assertAlmostEqual(euler.roll, self.roll)
        self.assertAlmostEqual(euler.pitch, self.pitch)
        self.assertAlmostEqual(euler.yaw, self.yaw)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        with self.assertRaises(AssertionError):
            EulerAngles.from_array(np.array([1, 2]))
        with self.assertRaises(AssertionError):
            EulerAngles.from_array(np.array([[1, 2, 3]]))

    def test_from_array_copy(self):
        """Test from_array with copy parameter."""
        original_array = self.test_array.copy()
        euler_copy = EulerAngles.from_array(original_array, copy=True)
        euler_no_copy = EulerAngles.from_array(original_array, copy=False)

        original_array[0] = 999.0
        self.assertNotEqual(euler_copy.roll, 999.0)
        self.assertEqual(euler_no_copy.roll, 999.0)

    def test_from_rotation_matrix(self):
        """Test from_rotation_matrix class method."""
        identity_matrix = np.eye(3)
        euler = EulerAngles.from_rotation_matrix(identity_matrix)
        self.assertAlmostEqual(euler.roll, 0.0, places=10)
        self.assertAlmostEqual(euler.pitch, 0.0, places=10)
        self.assertAlmostEqual(euler.yaw, 0.0, places=10)

    def test_from_rotation_matrix_invalid(self):
        """Test from_rotation_matrix with invalid input."""
        with self.assertRaises(AssertionError):
            EulerAngles.from_rotation_matrix(np.array([[1, 2]]))
        with self.assertRaises(AssertionError):
            EulerAngles.from_rotation_matrix(np.array([1, 2, 3]))

    def test_array_property(self):
        """Test array property."""
        array = self.euler_angles.array
        self.assertEqual(array.shape, (3,))
        self.assertEqual(array[EulerAnglesIndex.ROLL], self.roll)
        self.assertEqual(array[EulerAnglesIndex.PITCH], self.pitch)
        self.assertEqual(array[EulerAnglesIndex.YAW], self.yaw)


class TestQuaternion(unittest.TestCase):
    """Unit tests for Quaternion class."""

    def setUp(self):
        """Set up test fixtures."""
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.quaternion = Quaternion(self.qw, self.qx, self.qy, self.qz)
        self.test_array = np.zeros([4], dtype=np.float64)
        self.test_array[QuaternionIndex.QW] = self.qw
        self.test_array[QuaternionIndex.QX] = self.qx
        self.test_array[QuaternionIndex.QY] = self.qy
        self.test_array[QuaternionIndex.QZ] = self.qz

    def test_init(self):
        """Test Quaternion initialization."""
        quat = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.assertEqual(quat.qw, 1.0)
        self.assertEqual(quat.qx, 0.0)
        self.assertEqual(quat.qy, 0.0)
        self.assertEqual(quat.qz, 0.0)

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        quat = Quaternion.from_array(self.test_array)
        self.assertAlmostEqual(quat.qw, self.qw)
        self.assertAlmostEqual(quat.qx, self.qx)
        self.assertAlmostEqual(quat.qy, self.qy)
        self.assertAlmostEqual(quat.qz, self.qz)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        with self.assertRaises(AssertionError):
            Quaternion.from_array(np.array([1, 2, 3]))
        with self.assertRaises(AssertionError):
            Quaternion.from_array(np.array([[1, 2, 3, 4]]))

    def test_from_array_copy(self):
        """Test from_array with copy parameter."""
        original_array = self.test_array.copy()
        quat_copy = Quaternion.from_array(original_array, copy=True)
        quat_no_copy = Quaternion.from_array(original_array, copy=False)

        original_array[0] = 999.0
        self.assertNotEqual(quat_copy.qw, 999.0)
        self.assertEqual(quat_no_copy.qw, 999.0)

    def test_from_rotation_matrix(self):
        """Test from_rotation_matrix class method."""
        identity_matrix = np.eye(3)
        quat = Quaternion.from_rotation_matrix(identity_matrix)
        self.assertAlmostEqual(quat.qw, 1.0, places=10)
        self.assertAlmostEqual(quat.qx, 0.0, places=10)
        self.assertAlmostEqual(quat.qy, 0.0, places=10)
        self.assertAlmostEqual(quat.qz, 0.0, places=10)

    def test_from_rotation_matrix_invalid(self):
        """Test from_rotation_matrix with invalid input."""
        with self.assertRaises(AssertionError):
            Quaternion.from_rotation_matrix(np.array([[1, 2]]))
        with self.assertRaises(AssertionError):
            Quaternion.from_rotation_matrix(np.array([1, 2, 3]))

    def test_from_euler_angles(self):
        """Test from_euler_angles class method."""
        euler = EulerAngles(0.0, 0.0, 0.0)
        quat = Quaternion.from_euler_angles(euler)
        self.assertAlmostEqual(quat.qw, 1.0, places=10)
        self.assertAlmostEqual(quat.qx, 0.0, places=10)
        self.assertAlmostEqual(quat.qy, 0.0, places=10)
        self.assertAlmostEqual(quat.qz, 0.0, places=10)

    def test_array_property(self):
        """Test array property."""
        array = self.quaternion.array
        self.assertEqual(array.shape, (4,))
        np.testing.assert_array_equal(array, self.test_array)

    def test_pyquaternion_property(self):
        """Test pyquaternion property."""
        pyquat = self.quaternion.pyquaternion
        self.assertEqual(pyquat.w, self.qw)
        self.assertEqual(pyquat.x, self.qx)
        self.assertEqual(pyquat.y, self.qy)
        self.assertEqual(pyquat.z, self.qz)

    def test_euler_angles_property(self):
        """Test euler_angles property."""
        euler = self.quaternion.euler_angles
        self.assertIsInstance(euler, EulerAngles)
        self.assertAlmostEqual(euler.roll, 0.0, places=10)
        self.assertAlmostEqual(euler.pitch, 0.0, places=10)
        self.assertAlmostEqual(euler.yaw, 0.0, places=10)

    def test_rotation_matrix_property(self):
        """Test rotation_matrix property."""
        rot_matrix = self.quaternion.rotation_matrix
        self.assertEqual(rot_matrix.shape, (3, 3))
        np.testing.assert_array_almost_equal(rot_matrix, np.eye(3))


if __name__ == "__main__":
    unittest.main()
