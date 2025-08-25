from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from d123.common.utils.mixin import ArrayMixin
from d123.geometry.geometry_index import Point2DIndex, Point3DIndex


@dataclass
class Point2D(ArrayMixin):
    """Class to represents 2D points.

    :return: A Point2D instance.
    """

    x: float  # [m] location
    y: float  # [m] location
    __slots__ = "x", "y"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> Point2D:
        """Constructs a Point2D from a numpy array.

        :param array: Array of shape (2,) representing the point coordinates [x, y], indexed by \
            :class:`~d123.geometry.Point2DIndex`.
        :return: A Point2D instance.
        """

        assert array.ndim == 1
        assert array.shape[0] == len(Point2DIndex)
        return Point2D(array[Point2DIndex.X], array[Point2DIndex.Y])

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of the point.

        :return: A numpy array of shape (2,) containing the point coordinates [x, y], indexed by \
            :class:`~d123.geometry.Point2DIndex`.
        """
        array = np.zeros(len(Point2DIndex), dtype=np.float64)
        array[Point2DIndex.X] = self.x
        array[Point2DIndex.Y] = self.y
        return array

    @property
    def shapely_point(self) -> geom.Point:
        """The Shapely Point representation of the 2D point.

        :return: A Shapely Point representation of the 2D point.
        """
        return geom.Point(self.x, self.y)

    def __iter__(self) -> Iterable[float]:
        """Iterator over point coordinates."""
        return iter((self.x, self.y))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


@dataclass
class Point3D(ArrayMixin):
    """Class to represents 3D points."""

    x: float  # [m] location
    y: float  # [m] location
    z: float  # [m] location
    __slots__ = "x", "y", "z"

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> "Point3D":
        """Constructs a Point3D from a numpy array.

        :param array: Array of shape (3,) representing the point coordinates [x, y, z], indexed by \
            :class:`~d123.geometry.Point3DIndex`.
        :return: A Point3D instance.
        """
        assert array.ndim == 1, f"Array must be 1-dimensional, got shape {array.shape}"
        assert array.shape[0] == len(
            Point3DIndex
        ), f"Array must have the same length as Point3DIndex, got shape {array.shape}"
        return cls(array[Point3DIndex.X], array[Point3DIndex.Y], array[Point3DIndex.Z])

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the Point3D instance to a numpy array, indexed by :class:`~d123.geometry.Point3DIndex`.

        :return: A numpy array of shape (3,) containing the point coordinates [x, y, z].
        """
        array = np.zeros(len(Point3DIndex), dtype=np.float64)
        array[Point3DIndex.X] = self.x
        array[Point3DIndex.Y] = self.y
        array[Point3DIndex.Z] = self.z
        return array

    @property
    def point_2d(self) -> Point2D:
        """The 2D projection of the 3D point.

        :return: A Point2D instance representing the 2D projection of the 3D point.
        """
        return Point2D(self.x, self.y)

    @property
    def shapely_point(self) -> geom.Point:
        """The Shapely Point representation of the 3D point. \
            This geometry contains the z-coordinate, but many Shapely operations ignore it.

        :return: A Shapely Point representation of the 3D point.
        """
        return geom.Point(self.x, self.y, self.z)

    def __iter__(self) -> Iterable[float]:
        """Iterator over the point coordinates (x, y, z)."""
        return iter((self.x, self.y, self.z))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y, self.z))
