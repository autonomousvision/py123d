from __future__ import annotations

from functools import cached_property

import numpy as np
import numpy.typing as npt
import pyquaternion

from d123.common.utils.mixin import ArrayMixin
from d123.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex
from d123.geometry.utils.rotation_utils import get_rotation_matrix_from_euler_array


class EulerAngles(ArrayMixin):
    """Class to represent 3D rotation using Euler angles (roll, pitch, yaw) in radians.
    NOTE: The rotation order is intrinsic Z-Y'-X'' (yaw-pitch-roll).
    See https://en.wikipedia.org/wiki/Euler_angles for more details.
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, roll: float, pitch: float, yaw: float):
        """Initialize EulerAngles with roll, pitch, yaw coordinates."""
        array = np.zeros(len(EulerAnglesIndex), dtype=np.float64)
        array[EulerAnglesIndex.ROLL] = roll
        array[EulerAnglesIndex.PITCH] = pitch
        array[EulerAnglesIndex.YAW] = yaw
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> EulerAngles:
        """Constructs a EulerAngles from a numpy array.

        :param array: Array of shape (3,) representing the euler angles [roll, pitch, yaw], indexed by \
            :class:`~d123.geometry.EulerAnglesIndex`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A EulerAngles instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(EulerAnglesIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix: npt.NDArray[np.float64]) -> EulerAngles:
        """Constructs a EulerAngles from a 3x3 rotation matrix.
        NOTE: The rotation order is intrinsic Z-Y'-X'' (yaw-pitch-roll).

        :param rotation_matrix: A 3x3 numpy array representing the rotation matrix.
        :return: A EulerAngles instance.
        """
        assert rotation_matrix.ndim == 2
        assert rotation_matrix.shape == (3, 3)
        quaternion = pyquaternion.Quaternion(matrix=rotation_matrix)
        yaw, pitch, roll = quaternion.yaw_pitch_roll
        return EulerAngles(roll=roll, pitch=pitch, yaw=yaw)

    @property
    def roll(self) -> float:
        """The roll (x-axis rotation) angle in radians.

        :return: The roll angle in radians.
        """
        return self._array[EulerAnglesIndex.ROLL]

    @property
    def pitch(self) -> float:
        """The pitch (y-axis rotation) angle in radians.

        :return: The pitch angle in radians.
        """
        return self._array[EulerAnglesIndex.PITCH]

    @property
    def yaw(self) -> float:
        """The yaw (z-axis rotation) angle in radians.

        :return: The yaw angle in radians.
        """
        return self._array[EulerAnglesIndex.YAW]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the EulerAngles instance to a numpy array.

        :return: A numpy array of shape (3,) containing the Euler angles [roll, pitch, yaw], indexed by \
            :class:`~d123.geometry.EulerAnglesIndex`.
        """
        return self._array

    @cached_property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the Euler angles.
        NOTE: The rotation order is intrinsic Z-Y'-X'' (yaw-pitch-roll).

        :return: A 3x3 numpy array representing the rotation matrix.
        """
        return get_rotation_matrix_from_euler_array(self.array)

    def __iter__(self):
        """Iterator over euler angles."""
        return iter((self.roll, self.pitch, self.yaw))

    def __hash__(self):
        """Hash function for euler angles."""
        return hash((self.roll, self.pitch, self.yaw))


class Quaternion(ArrayMixin):
    """
    Represents a quaternion for 3D rotations.
    NOTE: This class uses the pyquaternion library for internal computations.
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, qw: float, qx: float, qy: float, qz: float):
        """Initialize Quaternion with qw, qx, qy, qz components."""
        array = np.zeros(len(QuaternionIndex), dtype=np.float64)
        array[QuaternionIndex.QW] = qw
        array[QuaternionIndex.QX] = qx
        array[QuaternionIndex.QY] = qy
        array[QuaternionIndex.QZ] = qz
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float64], copy: bool = True) -> Quaternion:
        """Constructs a Quaternion from a numpy array.

        :param arr: A 1D numpy array of shape (4,) containing the quaternion components [qw, qx, qy, qz].
        :param copy: Whether to copy the array data, defaults to True.
        :return: A Quaternion instance.
        """
        assert arr.ndim == 1
        assert arr.shape[0] == len(QuaternionIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", arr.copy() if copy else arr)
        return instance

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix: npt.NDArray[np.float64]) -> Quaternion:
        """Constructs a Quaternion from a 3x3 rotation matrix.

        :param rotation_matrix: A 3x3 numpy array representing the rotation matrix.
        :return: A Quaternion instance.
        """
        assert rotation_matrix.ndim == 2
        assert rotation_matrix.shape == (3, 3)
        quaternion = pyquaternion.Quaternion(matrix=rotation_matrix)
        return Quaternion(qw=quaternion.w, qx=quaternion.x, qy=quaternion.y, qz=quaternion.z)

    @classmethod
    def from_euler_angles(cls, euler_angles: EulerAngles) -> Quaternion:
        """Constructs a Quaternion from Euler angles.
        NOTE: The rotation order is intrinsic Z-Y'-X'' (yaw-pitch-roll).

        :param euler_angles: An EulerAngles instance representing the Euler angles.
        :return: A Quaternion instance.
        """
        rotation_matrix = euler_angles.rotation_matrix
        return Quaternion.from_rotation_matrix(rotation_matrix)

    @property
    def qw(self) -> float:
        """The scalar part of the quaternion.

        :return: The qw component.
        """
        return self._array[QuaternionIndex.QW]

    @property
    def qx(self) -> float:
        """The x component of the quaternion.

        :return: The qx component.
        """
        return self._array[QuaternionIndex.QX]

    @property
    def qy(self) -> float:
        """The y component of the quaternion.

        :return: The qy component.
        """
        return self._array[QuaternionIndex.QY]

    @property
    def qz(self) -> float:
        """The z component of the quaternion.

        :return: The qz component.
        """
        return self._array[QuaternionIndex.QZ]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the Quaternion instance to a numpy array.

        :return: A numpy array of shape (4,) containing the quaternion [qw, qx, qy, qz], indexed by \
            :class:`~d123.geometry.QuaternionIndex`.
        """
        return self._array

    @cached_property
    def pyquaternion(self) -> pyquaternion.Quaternion:
        """Returns the pyquaternion.Quaternion representation of the quaternion.

        :return: A pyquaternion.Quaternion representation of the quaternion.
        """
        return pyquaternion.Quaternion(array=self.array)

    @cached_property
    def euler_angles(self) -> EulerAngles:
        """Returns the Euler angles (roll, pitch, yaw) representation of the quaternion.
        NOTE: The rotation order is intrinsic Z-Y'-X'' (yaw-pitch-roll).

        :return: An EulerAngles instance representing the Euler angles.
        """
        yaw, pitch, roll = self.pyquaternion.yaw_pitch_roll
        return EulerAngles(roll=roll, pitch=pitch, yaw=yaw)

    @cached_property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the quaternion.

        :return: A 3x3 numpy array representing the rotation matrix.
        """
        return self.pyquaternion.rotation_matrix

    def __iter__(self):
        """Iterator over quaternion components."""
        return iter((self.qw, self.qx, self.qy, self.qz))

    def __hash__(self):
        """Hash function for quaternion."""
        return hash((self.qw, self.qx, self.qy, self.qz))
