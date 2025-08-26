from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom
from pyparsing import cached_property

from d123.common.utils.mixin import ArrayMixin
from d123.geometry.geometry_index import Point3DIndex, QuaternionSE3Index, StateSE2Index, StateSE3Index
from d123.geometry.point import Point2D, Point3D
from d123.geometry.rotation import EulerAngles, Quaternion


class StateSE2(ArrayMixin):
    """Class to represents a 2D pose as SE2 (x, y, yaw)."""

    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, yaw: float):
        """Initialize StateSE2 with x, y, yaw coordinates."""
        array = np.zeros(len(StateSE2Index), dtype=np.float64)
        array[StateSE2Index.X] = x
        array[StateSE2Index.Y] = y
        array[StateSE2Index.YAW] = yaw
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> StateSE2:
        """Constructs a StateSE2 from a numpy array.

        :param array: Array of shape (3,) representing the state [x, y, yaw], indexed by \
            :class:`~d123.geometry.geometry_index.StateSE2Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A StateSE2 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(StateSE2Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def x(self) -> float:
        return self._array[StateSE2Index.X]

    @property
    def y(self) -> float:
        return self._array[StateSE2Index.Y]

    @property
    def yaw(self) -> float:
        return self._array[StateSE2Index.YAW]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the StateSE2 instance to a numpy array

        :return: A numpy array of shape (3,) containing the state, indexed by \
            :class:`~d123.geometry.geometry_index.StateSE2Index`.
        """
        return self._array

    @property
    def state_se2(self) -> StateSE2:
        """The 2D pose itself. Helpful for polymorphism.

        :return: A StateSE2 instance representing the 2D pose.
        """
        return self

    @property
    def point_2d(self) -> Point2D:
        """The 2D projection of the 2D pose.

        :return: A Point2D instance representing the 2D projection of the 2D pose.
        """
        return Point2D.from_array(self.array[StateSE2Index.XY])

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 2x2 rotation matrix representation of the state's orientation.

        :return: A 2x2 numpy array representing the rotation matrix.
        """
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        return np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)

    @property
    def transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 transformation matrix representation of the state.

        :return: A 3x3 numpy array representing the transformation matrix.
        """
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[:2, :2] = self.rotation_matrix
        matrix[0, 2] = self.x
        matrix[1, 2] = self.y
        return matrix

    @property
    def shapely_point(self) -> geom.Point:
        return geom.Point(self.x, self.y)


class StateSE3(ArrayMixin):
    """
    Class to represents a 3D pose as SE3 (x, y, z, roll, pitch, yaw).
    TODO: Use quaternions for rotation representation.
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        """Initialize StateSE3 with x, y, z, roll, pitch, yaw coordinates."""
        array = np.zeros(len(StateSE3Index), dtype=np.float64)
        array[StateSE3Index.X] = x
        array[StateSE3Index.Y] = y
        array[StateSE3Index.Z] = z
        array[StateSE3Index.ROLL] = roll
        array[StateSE3Index.PITCH] = pitch
        array[StateSE3Index.YAW] = yaw
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> StateSE3:
        """Constructs a StateSE3 from a numpy array.

        :param array: Array of shape (6,) representing the state [x, y, z, roll, pitch, yaw], indexed by \
            :class:`~d123.geometry.geometry_index.StateSE3Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A StateSE3 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(StateSE3Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @classmethod
    def from_transformation_matrix(cls, array: npt.NDArray[np.float64]) -> StateSE3:
        """Constructs a StateSE3 from a 4x4 transformation matrix.

        :param array: A 4x4 numpy array representing the transformation matrix.
        :return: A StateSE3 instance.
        """
        assert array.ndim == 2
        assert array.shape == (4, 4)
        translation = array[:3, 3]
        rotation = array[:3, :3]
        roll, pitch, yaw = EulerAngles.from_rotation_matrix(rotation)

        return StateSE3(
            x=translation[Point3DIndex.X],
            y=translation[Point3DIndex.Y],
            z=translation[Point3DIndex.Z],
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

    @property
    def x(self) -> float:
        """Returns the x-coordinate of the 3D state.

        :return: The x-coordinate.
        """
        return self._array[StateSE3Index.X]

    @property
    def y(self) -> float:
        """Returns the y-coordinate of the 3D state.

        :return: The y-coordinate.
        """
        return self._array[StateSE3Index.Y]

    @property
    def z(self) -> float:
        """Returns the z-coordinate of the 3D state.

        :return: The z-coordinate.
        """
        return self._array[StateSE3Index.Z]

    @property
    def roll(self) -> float:
        """Returns the roll (x-axis rotation) of the 3D state.

        :return: The roll angle.
        """
        return self._array[StateSE3Index.ROLL]

    @property
    def pitch(self) -> float:
        """Returns the pitch (y-axis rotation) of the 3D state.

        :return: The pitch angle.
        """
        return self._array[StateSE3Index.PITCH]

    @property
    def yaw(self) -> float:
        """Returns the yaw (z-axis rotation) of the 3D state.

        :return: The yaw angle.
        """
        return self._array[StateSE3Index.YAW]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Returns the StateSE3 instance as a numpy array.

        :return: A numpy array of shape (6,), indexed by \
            :class:`~d123.geometry.geometry_index.StateSE3Index`.
        """
        return self._array

    @property
    def state_se2(self) -> StateSE2:
        """Returns the 3D state as a 2D state by ignoring the z-axis.

        :return: A StateSE2 instance representing the 2D projection of the 3D state.
        """
        return StateSE2(self.x, self.y, self.yaw)

    @property
    def point_3d(self) -> Point3D:
        """Returns the 3D point representation of the state.

        :return: A Point3D instance representing the 3D point.
        """
        return Point3D(self.x, self.y, self.z)

    @property
    def point_2d(self) -> Point2D:
        """Returns the 2D point representation of the state.

        :return: A Point2D instance representing the 2D point.
        """
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        """Returns the Shapely point representation of the state.

        :return: A Shapely Point instance representing the 3D point.
        """
        return self.point_3d.shapely_point

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the state's orientation.

        :return: A 3x3 numpy array representing the rotation matrix.
        """
        return self.euler_angles.rotation_matrix

    @property
    def transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 4x4 transformation matrix representation of the state.

        :return: A 4x4 numpy array representing the transformation matrix.
        """
        rotation_matrix = self.rotation_matrix
        transformation_matrix = np.eye(4, dtype=np.float64)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = self.array[StateSE3Index.XYZ]
        return transformation_matrix

    @cached_property
    def euler_angles(self) -> EulerAngles:
        return EulerAngles.from_array(self.array[StateSE3Index.EULER_ANGLES])

    @property
    def quaternion(self) -> npt.NDArray[np.float64]:
        """Returns the quaternion (w, x, y, z) representation of the state's orientation.

        :return: A numpy array of shape (4,) representing the quaternion.
        """
        raise NotImplementedError("Quaternion conversion not implemented yet.")

    def __iter__(self) -> Iterable[float]:
        """Iterator over the state coordinates (x, y, z, roll, pitch, yaw)."""
        return iter((self.x, self.y, self.z, self.roll, self.pitch, self.yaw))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y, self.z, self.roll, self.pitch, self.yaw))

    def __matmul__(self, other: StateSE3) -> StateSE3:
        """Combines two SE3 states by applying the transformation of the other state to this state.

        :param other: Another StateSE3 instance representing the transformation to apply.
        :return: A new StateSE3 instance representing the combined transformation.
        """
        return StateSE3.from_transformation_matrix(self.transformation_matrix @ other.transformation_matrix)


class QuaternionSE3:
    """Class representing a quaternion in SE3 space.

    TODO: Implement and replace StateSE3.
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float):
        """Initialize QuaternionSE3 with x, y, z, qw, qx, qy, qz coordinates."""
        array = np.zeros(len(QuaternionSE3Index), dtype=np.float64)
        array[QuaternionSE3Index.X] = x
        array[QuaternionSE3Index.Y] = y
        array[QuaternionSE3Index.Z] = z
        array[QuaternionSE3Index.QW] = qw
        array[QuaternionSE3Index.QX] = qx
        array[QuaternionSE3Index.QY] = qy
        array[QuaternionSE3Index.QZ] = qz
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> QuaternionSE3:
        """Constructs a QuaternionSE3 from a numpy array.

        :param array: Array of shape (7,), indexed by :class:`~d123.geometry.geometry_index.QuaternionSE3Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A QuaternionSE3 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(QuaternionSE3Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def x(self) -> float:
        """Returns the x-coordinate of the quaternion.

        :return: The x-coordinate.
        """
        return self._array[QuaternionSE3Index.X]

    @property
    def y(self) -> float:
        """Returns the y-coordinate of the quaternion.

        :return: The y-coordinate.
        """
        return self._array[QuaternionSE3Index.Y]

    @property
    def z(self) -> float:
        """Returns the z-coordinate of the quaternion.

        :return: The z-coordinate.
        """
        return self._array[QuaternionSE3Index.Z]

    @property
    def qw(self) -> float:
        """Returns the w-coordinate of the quaternion.

        :return: The w-coordinate.
        """
        return self._array[QuaternionSE3Index.QW]

    @property
    def qx(self) -> float:
        """Returns the x-coordinate of the quaternion.

        :return: The x-coordinate.
        """
        return self._array[QuaternionSE3Index.QX]

    @property
    def qy(self) -> float:
        """Returns the y-coordinate of the quaternion.

        :return: The y-coordinate.
        """
        return self._array[QuaternionSE3Index.QY]

    @property
    def qz(self) -> float:
        """Returns the z-coordinate of the quaternion.

        :return: The z-coordinate.
        """
        return self._array[QuaternionSE3Index.QZ]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the QuaternionSE3 instance to a numpy array.

        :return: A numpy array of shape (7,), indexed by :class:`~d123.geometry.geometry_index.QuaternionSE3Index`.
        """
        return self._array

    @property
    def state_se2(self) -> StateSE2:
        """Returns the quaternion state as a 2D state by ignoring the z-axis.

        :return: A StateSE2 instance representing the 2D projection of the 3D state.
        """
        # Convert quaternion to yaw angle
        yaw = self.quaternion.euler_angles.yaw
        return StateSE2(self.x, self.y, yaw)

    @property
    def point_3d(self) -> Point3D:
        """Returns the 3D point representation of the state.

        :return: A Point3D instance representing the 3D point.
        """
        return Point3D(self.x, self.y, self.z)

    @property
    def point_2d(self) -> Point2D:
        """Returns the 2D point representation of the state.

        :return: A Point2D instance representing the 2D point.
        """
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        """Returns the Shapely point representation of the state.

        :return: A Shapely Point instance representing the 3D point.
        """
        return self.point_3d.shapely_point

    @cached_property
    def quaternion(self) -> Quaternion:
        """Returns the quaternion (w, x, y, z) representation of the state's orientation.

        :return: A Quaternion instance representing the quaternion.
        """
        return Quaternion.from_array(self.array[QuaternionSE3Index.QUATERNION])
