from __future__ import annotations

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry.geometry_index import EulerStateSE3Index, Point3DIndex, PoseSE2Index, PoseSE3Index
from py123d.geometry.point import Point2D, Point3D
from py123d.geometry.rotation import EulerAngles, Quaternion


class PoseSE2(ArrayMixin):
    """Class to represents a 2D pose as SE2 (x, y, yaw).

    Examples:
        >>> from py123d.geometry import PoseSE2
        >>> pose = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        >>> print(pose.x, pose.y, pose.yaw)
        1.0 2.0 0.5
        >>> print(pose.rotation_matrix)
        [[ 0.87758256 -0.47942554]
         [ 0.47942554  0.87758256]]

    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, yaw: float):
        """Init :class:`PoseSE2` with x, y, yaw coordinates.

        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param yaw: The yaw angle in radians.
        """
        array = np.zeros(len(PoseSE2Index), dtype=np.float64)
        array[PoseSE2Index.X] = x
        array[PoseSE2Index.Y] = y
        array[PoseSE2Index.YAW] = yaw
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PoseSE2:
        """Constructs a PoseSE2 from a numpy array.

        :param array: Array of shape (3,) representing the state [x, y, yaw], indexed by \
            :class:`~py123d.geometry.geometry_index.PoseSE2Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A PoseSE2 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(PoseSE2Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def x(self) -> float:
        """The x-coordinate of the pose."""
        return self._array[PoseSE2Index.X]

    @property
    def y(self) -> float:
        """The y-coordinate of the pose."""
        return self._array[PoseSE2Index.Y]

    @property
    def yaw(self) -> float:
        """The yaw angle of the pose."""
        return self._array[PoseSE2Index.YAW]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Pose as numpy array of shape (3,), indexed by :class:`~py123d.geometry.geometry_index.PoseSE2Index`."""
        return self._array

    @property
    def point_2d(self) -> Point2D:
        """The :class:`~py123d.geometry.Point2D` of the pose, i.e. the translation part."""
        return Point2D.from_array(self.array[PoseSE2Index.XY])

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """The 2x2 rotation matrix representation of the pose."""
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        return np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)

    @property
    def transformation_matrix(self) -> npt.NDArray[np.float64]:
        """The 3x3 transformation matrix representation of the pose."""
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[:2, :2] = self.rotation_matrix
        matrix[0, 2] = self.x
        matrix[1, 2] = self.y
        return matrix

    @property
    def shapely_point(self) -> geom.Point:
        """The Shapely point representation of the pose."""
        return geom.Point(self.x, self.y)

    @property
    def pose_se2(self) -> PoseSE2:
        """Returns self to match interface of other pose classes."""
        return self


class PoseSE3(ArrayMixin):
    """Class representing a quaternion in SE3 space

    Examples:
        >>> from py123d.geometry import PoseSE3
        >>> pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        >>> pose.point_3d
        Point3D(array=[1. 2. 3.])
        >>> pose.transformation_matrix
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])
        >>> PoseSE3.from_transformation_matrix(pose.transformation_matrix) == pose
        True
        >>> print(pose.yaw, pose.pitch, pose.roll)
        0.0 0.0 0.0
    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float):
        """Initialize :class:`PoseSE3` with x, y, z, qw, qx, qy, qz coordinates.

        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param z: The z-coordinate.
        :param qw: The w-coordinate of the quaternion, representing the scalar part.
        :param qx: The x-coordinate of the quaternion, representing the first component of the vector part.
        :param qy: The y-coordinate of the quaternion, representing the second component of the vector part.
        :param qz: The z-coordinate of the quaternion, representing the third component of the vector part.
        """
        array = np.zeros(len(PoseSE3Index), dtype=np.float64)
        array[PoseSE3Index.X] = x
        array[PoseSE3Index.Y] = y
        array[PoseSE3Index.Z] = z
        array[PoseSE3Index.QW] = qw
        array[PoseSE3Index.QX] = qx
        array[PoseSE3Index.QY] = qy
        array[PoseSE3Index.QZ] = qz
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PoseSE3:
        """Constructs a :class:`PoseSE3` from a numpy array of shape (7,), \
            indexed by :class:`~py123d.geometry.geometry_index.PoseSE3Index`.

        :param array: Array of shape (7,) representing the state [x, y, z, qw, qx, qy, qz].
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A :class:`PoseSE3` instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(PoseSE3Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @classmethod
    def from_transformation_matrix(cls, transformation_matrix: npt.NDArray[np.float64]) -> PoseSE3:
        """Constructs a :class:`PoseSE3` from a 4x4 transformation matrix.

        :param transformation_matrix: A 4x4 numpy array representing the transformation matrix.
        :return: A :class:`PoseSE3` instance.
        """
        assert transformation_matrix.ndim == 2
        assert transformation_matrix.shape == (4, 4)
        array = np.zeros(len(PoseSE3Index), dtype=np.float64)
        array[PoseSE3Index.XYZ] = transformation_matrix[:3, 3]
        array[PoseSE3Index.QUATERNION] = Quaternion.from_rotation_matrix(transformation_matrix[:3, :3])
        return PoseSE3.from_array(array, copy=False)

    @property
    def x(self) -> float:
        """The x-coordinate of the pose."""
        return self._array[PoseSE3Index.X]

    @property
    def y(self) -> float:
        """The y-coordinate of the pose."""
        return self._array[PoseSE3Index.Y]

    @property
    def z(self) -> float:
        """The z-coordinate of the pose."""
        return self._array[PoseSE3Index.Z]

    @property
    def qw(self) -> float:
        """The w-coordinate of the quaternion, representing the scalar part."""
        return self._array[PoseSE3Index.QW]

    @property
    def qx(self) -> float:
        """The x-coordinate of the quaternion, representing the first component of the vector part."""
        return self._array[PoseSE3Index.QX]

    @property
    def qy(self) -> float:
        """The y-coordinate of the quaternion, representing the second component of the vector part."""
        return self._array[PoseSE3Index.QY]

    @property
    def qz(self) -> float:
        """The z-coordinate of the quaternion, representing the third component of the vector part."""
        return self._array[PoseSE3Index.QZ]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of the pose with shape (7,), \
            indexed by :class:`~py123d.geometry.geometry_index.PoseSE3Index`"""
        return self._array

    @property
    def pose_se2(self) -> PoseSE2:
        """The :class:`PoseSE2` representation of the SE3 pose."""
        return PoseSE2(self.x, self.y, self.yaw)

    @property
    def point_3d(self) -> Point3D:
        """The :class:`Point3D` representation of the SE3 pose, i.e. the translation part."""
        return Point3D(self.x, self.y, self.z)

    @property
    def point_2d(self) -> Point2D:
        """The :class:`Point2D` representation of the SE3 pose, i.e. the translation part."""
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        """The Shapely point representation, of the translation part of the SE3 pose."""
        return self.point_3d.shapely_point

    @property
    def quaternion(self) -> Quaternion:
        """The :class:`~py123d.geometry.Quaternion` representation of the state's orientation."""
        return Quaternion.from_array(self.array[PoseSE3Index.QUATERNION])

    @property
    def euler_angles(self) -> EulerAngles:
        """The :class:`~py123d.geometry.EulerAngles` representation of the state's orientation."""
        return self.quaternion.euler_angles

    @property
    def roll(self) -> float:
        """The roll (x-axis rotation) angle in radians."""
        return self.euler_angles.roll

    @property
    def pitch(self) -> float:
        """The pitch (y-axis rotation) angle in radians."""
        return self.euler_angles.pitch

    @property
    def yaw(self) -> float:
        """The yaw (z-axis rotation) angle in radians."""
        return self.euler_angles.yaw

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the state's orientation."""
        return self.quaternion.rotation_matrix

    @property
    def transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 4x4 transformation matrix representation of the state."""
        transformation_matrix = np.eye(4, dtype=np.float64)
        transformation_matrix[:3, :3] = self.rotation_matrix
        transformation_matrix[:3, 3] = self.array[PoseSE3Index.XYZ]
        return transformation_matrix


class EulerStateSE3(ArrayMixin):
    """
    Class to represents a 3D pose as SE3 (x, y, z, roll, pitch, yaw).

    Notes
    -----
    This class is deprecated, use :class:`~py123d.geometry.PoseSE3` instead (quaternion based).
    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        """Initialize PoseSE3 with x, y, z, roll, pitch, yaw coordinates."""
        array = np.zeros(len(EulerStateSE3Index), dtype=np.float64)
        array[EulerStateSE3Index.X] = x
        array[EulerStateSE3Index.Y] = y
        array[EulerStateSE3Index.Z] = z
        array[EulerStateSE3Index.ROLL] = roll
        array[EulerStateSE3Index.PITCH] = pitch
        array[EulerStateSE3Index.YAW] = yaw
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> EulerStateSE3:
        """Constructs a PoseSE3 from a numpy array.

        :param array: Array of shape (6,) representing the state [x, y, z, roll, pitch, yaw], indexed by \
            :class:`~py123d.geometry.geometry_index.PoseSE3Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A PoseSE3 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(EulerStateSE3Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @classmethod
    def from_transformation_matrix(cls, transformation_matrix: npt.NDArray[np.float64]) -> EulerStateSE3:
        """Constructs a EulerStateSE3 from a 4x4 transformation matrix.

        :param array: A 4x4 numpy array representing the transformation matrix.
        :return: A EulerStateSE3 instance.
        """
        assert transformation_matrix.ndim == 2
        assert transformation_matrix.shape == (4, 4)
        translation = transformation_matrix[:3, 3]
        rotation = transformation_matrix[:3, :3]
        roll, pitch, yaw = EulerAngles.from_rotation_matrix(rotation)
        return EulerStateSE3(
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
        return self._array[EulerStateSE3Index.X]

    @property
    def y(self) -> float:
        """Returns the y-coordinate of the 3D state.

        :return: The y-coordinate.
        """
        return self._array[EulerStateSE3Index.Y]

    @property
    def z(self) -> float:
        """Returns the z-coordinate of the 3D state.

        :return: The z-coordinate.
        """
        return self._array[EulerStateSE3Index.Z]

    @property
    def roll(self) -> float:
        """Returns the roll (x-axis rotation) of the 3D state.

        :return: The roll angle.
        """
        return self._array[EulerStateSE3Index.ROLL]

    @property
    def pitch(self) -> float:
        """Returns the pitch (y-axis rotation) of the 3D state.

        :return: The pitch angle.
        """
        return self._array[EulerStateSE3Index.PITCH]

    @property
    def yaw(self) -> float:
        """Returns the yaw (z-axis rotation) of the 3D state.

        :return: The yaw angle.
        """
        return self._array[EulerStateSE3Index.YAW]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Returns the PoseSE3 instance as a numpy array.

        :return: A numpy array of shape (6,), indexed by \
            :class:`~py123d.geometry.geometry_index.PoseSE3Index`.
        """
        return self._array

    @property
    def pose_se2(self) -> PoseSE2:
        """Returns the 3D state as a 2D state by ignoring the z-axis.

        :return: A StateSE2 instance representing the 2D projection of the 3D state.
        """
        return PoseSE2(self.x, self.y, self.yaw)

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
        transformation_matrix[:3, 3] = self.array[EulerStateSE3Index.XYZ]
        return transformation_matrix

    @property
    def euler_angles(self) -> EulerAngles:
        return EulerAngles.from_array(self.array[EulerStateSE3Index.EULER_ANGLES])

    @property
    def pose_se3(self) -> PoseSE3:
        quaternion_se3_array = np.zeros(len(PoseSE3Index), dtype=np.float64)
        quaternion_se3_array[PoseSE3Index.XYZ] = self.array[EulerStateSE3Index.XYZ]
        quaternion_se3_array[PoseSE3Index.QUATERNION] = Quaternion.from_euler_angles(self.euler_angles)
        return PoseSE3.from_array(quaternion_se3_array, copy=False)

    @property
    def quaternion(self) -> Quaternion:
        return Quaternion.from_euler_angles(self.euler_angles)
