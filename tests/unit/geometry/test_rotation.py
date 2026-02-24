import numpy as np
import pytest

from py123d.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex
from py123d.geometry.rotation import EulerAngles, Quaternion


class TestEulerAngles:
    """Unit tests for EulerAngles class."""

    def setup_method(self):
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
        assert euler.roll == 0.1
        assert euler.pitch == 0.2
        assert euler.yaw == 0.3

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        euler = EulerAngles.from_array(self.test_array)
        assert isinstance(euler, EulerAngles)
        assert euler.roll == pytest.approx(self.roll)
        assert euler.pitch == pytest.approx(self.pitch)
        assert euler.yaw == pytest.approx(self.yaw)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        with pytest.raises(AssertionError):
            EulerAngles.from_array(np.array([1, 2]))
        with pytest.raises(AssertionError):
            EulerAngles.from_array(np.array([[1, 2, 3]]))

    def test_from_array_copy(self):
        """Test from_array with copy parameter."""
        original_array = self.test_array.copy()
        euler_copy = EulerAngles.from_array(original_array, copy=True)
        euler_no_copy = EulerAngles.from_array(original_array, copy=False)

        original_array[0] = 999.0
        assert euler_copy.roll != 999.0
        assert euler_no_copy.roll == 999.0

    def test_from_rotation_matrix(self):
        """Test from_rotation_matrix class method."""
        identity_matrix = np.eye(3)
        euler = EulerAngles.from_rotation_matrix(identity_matrix)
        assert euler.roll == pytest.approx(0.0, abs=1e-10)
        assert euler.pitch == pytest.approx(0.0, abs=1e-10)
        assert euler.yaw == pytest.approx(0.0, abs=1e-10)

    def test_from_rotation_matrix_invalid(self):
        """Test from_rotation_matrix with invalid input."""
        with pytest.raises(AssertionError):
            EulerAngles.from_rotation_matrix(np.array([[1, 2]]))
        with pytest.raises(AssertionError):
            EulerAngles.from_rotation_matrix(np.array([1, 2, 3]))

    def test_array_property(self):
        """Test array property."""
        array = self.euler_angles.array
        assert array.shape == (3,)
        assert array[EulerAnglesIndex.ROLL] == self.roll
        assert array[EulerAnglesIndex.PITCH] == self.pitch
        assert array[EulerAnglesIndex.YAW] == self.yaw

    def test_repr(self):
        """Test __repr__ returns a string containing the class name."""
        r = repr(self.euler_angles)
        assert "EulerAngles" in r


class TestQuaternion:
    """Unit tests for Quaternion class."""

    def setup_method(self):
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
        assert quat.qw == 1.0
        assert quat.qx == 0.0
        assert quat.qy == 0.0
        assert quat.qz == 0.0

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        quat = Quaternion.from_array(self.test_array)
        assert quat.qw == pytest.approx(self.qw)
        assert quat.qx == pytest.approx(self.qx)
        assert quat.qy == pytest.approx(self.qy)
        assert quat.qz == pytest.approx(self.qz)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        with pytest.raises(AssertionError):
            Quaternion.from_array(np.array([1, 2, 3]))
        with pytest.raises(AssertionError):
            Quaternion.from_array(np.array([[1, 2, 3, 4]]))

    def test_from_array_copy(self):
        """Test from_array with copy parameter."""
        original_array = self.test_array.copy()
        quat_copy = Quaternion.from_array(original_array, copy=True)
        quat_no_copy = Quaternion.from_array(original_array, copy=False)

        original_array[0] = 999.0
        assert quat_copy.qw != 999.0
        assert quat_no_copy.qw == 999.0

    def test_from_rotation_matrix(self):
        """Test from_rotation_matrix class method."""
        identity_matrix = np.eye(3)
        quat = Quaternion.from_rotation_matrix(identity_matrix)
        assert quat.qw == pytest.approx(1.0, abs=1e-10)
        assert quat.qx == pytest.approx(0.0, abs=1e-10)
        assert quat.qy == pytest.approx(0.0, abs=1e-10)
        assert quat.qz == pytest.approx(0.0, abs=1e-10)

    def test_from_rotation_matrix_invalid(self):
        """Test from_rotation_matrix with invalid input."""
        with pytest.raises(AssertionError):
            Quaternion.from_rotation_matrix(np.array([[1, 2]]))
        with pytest.raises(AssertionError):
            Quaternion.from_rotation_matrix(np.array([1, 2, 3]))

    def test_from_euler_angles(self):
        """Test from_euler_angles class method."""
        euler = EulerAngles(0.0, 0.0, 0.0)
        quat = Quaternion.from_euler_angles(euler)
        assert quat.qw == pytest.approx(1.0, abs=1e-10)
        assert quat.qx == pytest.approx(0.0, abs=1e-10)
        assert quat.qy == pytest.approx(0.0, abs=1e-10)
        assert quat.qz == pytest.approx(0.0, abs=1e-10)

    def test_array_property(self):
        """Test array property."""
        array = self.quaternion.array
        assert array.shape == (4,)
        np.testing.assert_array_equal(array, self.test_array)

    def test_pyquaternion_property(self):
        """Test pyquaternion property."""
        pyquat = self.quaternion.pyquaternion
        assert pyquat.w == self.qw
        assert pyquat.x == self.qx
        assert pyquat.y == self.qy
        assert pyquat.z == self.qz

    def test_euler_angles_property(self):
        """Test euler_angles property."""
        euler = self.quaternion.euler_angles
        assert isinstance(euler, EulerAngles)
        assert euler.roll == pytest.approx(0.0, abs=1e-10)
        assert euler.pitch == pytest.approx(0.0, abs=1e-10)
        assert euler.yaw == pytest.approx(0.0, abs=1e-10)

    def test_rotation_matrix_property(self):
        """Test rotation_matrix property."""
        rot_matrix = self.quaternion.rotation_matrix
        assert rot_matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(rot_matrix, np.eye(3))

    def test_repr(self):
        """Test __repr__ returns a string containing the class name."""
        r = repr(self.quaternion)
        assert "Quaternion" in r
