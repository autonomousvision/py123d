from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from d123.geometry.geometry_index import Point2DIndex, Point3DIndex


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
