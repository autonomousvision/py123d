from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from d123.geometry.geometry_index import StateSE2Index, StateSE3Index
from d123.geometry.point import Point2D, Point3D


@dataclass
class StateSE2:
    """Class to represents a 2D pose as SE2 (x, y, yaw)."""

    x: float  # [m] x-location
    y: float  # [m] y-location
    yaw: float  # [rad] yaw/heading
    __slots__ = "x", "y", "yaw"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> StateSE2:
        """Constructs a StateSE2 from a numpy array.

        :param array: Array of shape (3,) representing the state [x, y, yaw], indexed by \
            :class:`~d123.geometry.geometry_index.StateSE2Index`.
        :return: A StateSE2 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(StateSE2Index)
        return StateSE2(array[StateSE2Index.X], array[StateSE2Index.Y], array[StateSE2Index.YAW])

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the StateSE2 instance to a numpy array, indexed by \
            :class:`~d123.geometry.geometry_index.StateSE2Index`.

        :return: A numpy array of shape (3,) containing the state [x, y, yaw].
        """
        array = np.zeros(len(StateSE2Index), dtype=np.float64)
        array[StateSE2Index.X] = self.x
        array[StateSE2Index.Y] = self.y
        array[StateSE2Index.YAW] = self.yaw
        return array

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
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        return geom.Point(self.x, self.y)

    def __iter__(self) -> Iterable[float]:
        """Iterator over the state coordinates (x, y, yaw)."""
        return iter((self.x, self.y, self.yaw))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y, self.yaw))


@dataclass
class StateSE3:
    """
    Class to represents a 3D pose as SE3 (x, y, z, roll, pitch, yaw).
    TODO: Use quaternions for rotation representation.
    """

    x: float  # [m] x-location
    y: float  # [m] y-location
    z: float  # [m] z-location
    roll: float  # [rad] roll
    pitch: float  # [rad] pitch
    yaw: float  # [rad] yaw
    __slots__ = "x", "y", "z", "roll", "pitch", "yaw"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> StateSE3:
        """Constructs a StateSE3 from a numpy array.

        :param array: Array of shape (6,) representing the state [x, y, z, roll, pitch, yaw], indexed by \
            :class:`~d123.geometry.StateSE3Index`.
        :return: A StateSE3 instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(StateSE3Index)
        return StateSE3(
            array[StateSE3Index.X],
            array[StateSE3Index.Y],
            array[StateSE3Index.Z],
            array[StateSE3Index.ROLL],
            array[StateSE3Index.PITCH],
            array[StateSE3Index.YAW],
        )

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
        return StateSE3(
            x=translation[0],
            y=translation[1],
            z=translation[2],
            roll=np.arctan2(rotation[2, 1], rotation[2, 2]),
            pitch=np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)),
            yaw=np.arctan2(rotation[1, 0], rotation[0, 0]),
        )

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the StateSE3 instance to a numpy array, indexed by StateSE3Index.

        :return: A numpy array of shape (6,) containing the state [x, y, z, roll, pitch, yaw].
        """
        array = np.zeros(len(StateSE3Index), dtype=np.float64)
        array[StateSE3Index.X] = self.x
        array[StateSE3Index.Y] = self.y
        array[StateSE3Index.Z] = self.z
        array[StateSE3Index.ROLL] = self.roll
        array[StateSE3Index.PITCH] = self.pitch
        array[StateSE3Index.YAW] = self.yaw
        return array

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
    def transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 4x4 transformation matrix representation of the state.

        :return: A 4x4 numpy array representing the transformation matrix.
        """
        raise NotImplementedError("Transformation matrix conversion not implemented yet.")

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the state's orientation.

        :return: A 3x3 numpy array representing the rotation matrix.
        """
        raise NotImplementedError("Rotation matrix conversion not implemented yet.")

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


@dataclass
class QuaternionSE3:
    """Class representing a quaternion in SE3 space.

    TODO: Implement and replace StateSE3.
    """

    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
