from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

# from d123.geometry.transform.se3 import get_rotation_matrix
from d123.common.utils.enums import classproperty

# TODO: Reconsider if 2D/3D or SE2/SE3 structure would be better hierarchical, e.g. inheritance or composition.


class Point2DIndex(IntEnum):
    X = 0
    Y = 1

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


@dataclass
class Point2D:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location
    __slots__ = "x", "y"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> Point2D:
        assert array.ndim == 1
        assert array.shape[0] == len(Point2DIndex)
        return Point2D(array[Point2DIndex.X], array[Point2DIndex.Y])

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        array = np.zeros(len(Point2DIndex), dtype=np.float64)
        array[Point2DIndex.X] = self.x
        array[Point2DIndex.Y] = self.y
        return array

    @property
    def shapely_point(self) -> geom.Point:
        return geom.Point(self.x, self.y)

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y)
        """
        return iter((self.x, self.y))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


class StateSE2Index(IntEnum):
    X = 0
    Y = 1
    YAW = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


@dataclass
class StateSE2:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location
    yaw: float  # [m] location
    __slots__ = "x", "y", "yaw"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> StateSE2:
        assert array.ndim == 1
        assert array.shape[0] == len(StateSE2Index)
        return StateSE2(array[StateSE2Index.X], array[StateSE2Index.Y], array[StateSE2Index.YAW])

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        array = np.zeros(len(StateSE2Index), dtype=np.float64)
        array[StateSE2Index.X] = self.x
        array[StateSE2Index.Y] = self.y
        array[StateSE2Index.YAW] = self.yaw
        return array

    @property
    def point_2d(self) -> Point2D:
        """
        Convert SE2 state to 2D point (drops heading)
        :return: Point2D dataclass
        """
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        return geom.Point(self.x, self.y)

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y)
        """
        return iter((self.x, self.y))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


class Point3DIndex(IntEnum):

    X = 0
    Y = 1
    Z = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


@dataclass
class Point3D:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location
    z: float  # [m] location
    __slots__ = "x", "y", "z"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> "Point3D":
        assert array.ndim == 1, f"Array must be 1-dimensional, got shape {array.shape}"
        assert array.shape[0] == len(
            Point3DIndex
        ), f"Array must have the same length as Point3DIndex, got shape {array.shape}"
        return cls(array[Point3DIndex.X], array[Point3DIndex.Y], array[Point3DIndex.Z])

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        array = np.zeros(len(Point3DIndex), dtype=np.float64)
        array[Point3DIndex.X] = self.x
        array[Point3DIndex.Y] = self.y
        array[Point3DIndex.Z] = self.z
        return array

    @property
    def point_2d(self) -> Point2D:
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        return geom.Point(self.x, self.y, self.z)

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y)
        """
        return iter((self.x, self.y, self.z))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y, self.z))


class StateSE3Index(IntEnum):

    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def ROTATION_XYZ(cls) -> slice:
        return slice(cls.ROLL, cls.YAW + 1)


@dataclass
class StateSE3:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location
    z: float  # [m] location
    roll: float
    pitch: float
    yaw: float
    __slots__ = "x", "y", "z", "roll", "pitch", "yaw"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> StateSE3:
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
    def from_matrix(cls, array: npt.NDArray[np.float64]) -> StateSE3:
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
        array = np.zeros(len(StateSE3Index), dtype=np.float64)
        array[StateSE3Index.X] = self.x
        array[StateSE3Index.Y] = self.y
        array[StateSE3Index.Z] = self.z
        array[StateSE3Index.ROLL] = self.roll
        array[StateSE3Index.PITCH] = self.pitch
        array[StateSE3Index.YAW] = self.yaw
        return array

    # @property
    # def matrix(self) -> npt.NDArray[np.float64]:
    #     """Convert SE3 state to 4x4 transformation matrix."""
    #     R = get_rotation_matrix(self)
    #     translation = np.array([self.x, self.y, self.z], dtype=np.float64)
    #     matrix = np.eye(4, dtype=np.float64)
    #     matrix[:3, :3] = R
    #     matrix[:3, 3] = translation
    #     return matrix

    @property
    def state_se2(self) -> StateSE2:
        return StateSE2(self.x, self.y, self.yaw)

    @property
    def point_3d(self) -> Point3D:
        return Point3D(self.x, self.y, self.z)

    @property
    def point_2d(self) -> Point2D:
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        return geom.Point(self.x, self.y, self.z)
