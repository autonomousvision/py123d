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
