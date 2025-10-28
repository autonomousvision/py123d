from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry.geometry_index import Point2DIndex, Point3DIndex


class Point2D(ArrayMixin):
    """Class to represents 2D points."""

    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float):
        """Initialize StateSE2 with x, y, yaw coordinates."""
        array = np.zeros(len(Point2DIndex), dtype=np.float64)
        array[Point2DIndex.X] = x
        array[Point2DIndex.Y] = y
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Point2D:
        """Constructs a Point2D from a numpy array.

        :param array: Array of shape (2,) representing the point coordinates [x, y], indexed by \
            :class:`~py123d.geometry.Point2DIndex`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A Point2D instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(Point2DIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def x(self) -> float:
        """The x coordinate of the point.

        :return: The x coordinate of the point.
        """
        return self._array[Point2DIndex.X]

    @property
    def y(self) -> float:
        """The y coordinate of the point.

        :return: The y coordinate of the point.
        """
        return self._array[Point2DIndex.Y]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of the point.

        :return: A numpy array of shape (2,) containing the point coordinates [x, y], indexed by \
            :class:`~py123d.geometry.Point2DIndex`.
        """
        return self._array

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


class Point3D(ArrayMixin):
    """Class to represents 3D points."""

    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float):
        """Initialize Point3D with x, y, z coordinates."""
        array = np.zeros(len(Point3DIndex), dtype=np.float64)
        array[Point3DIndex.X] = x
        array[Point3DIndex.Y] = y
        array[Point3DIndex.Z] = z
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Point3D:
        """Constructs a Point3D from a numpy array.

        :param array: Array of shape (3,) representing the point coordinates [x, y, z], indexed by \
            :class:`~py123d.geometry.Point3DIndex`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A Point3D instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(Point3DIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of the point.

        :return: A numpy array of shape (3,) containing the point coordinates [x, y, z], indexed by \
            :class:`~py123d.geometry.Point3DIndex`.
        """
        return self._array

    @property
    def x(self) -> float:
        """The x coordinate of the point.

        :return: The x coordinate of the point.
        """
        return self._array[Point3DIndex.X]

    @property
    def y(self) -> float:
        """The y coordinate of the point.

        :return: The y coordinate of the point.
        """
        return self._array[Point3DIndex.Y]

    @property
    def z(self) -> float:
        """The z coordinate of the point.

        :return: The z coordinate of the point.
        """
        return self._array[Point3DIndex.Z]

    @property
    def point_2d(self) -> Point2D:
        """The 2D projection of the 3D point.

        :return: A Point2D instance representing the 2D projection of the 3D point.
        """
        return Point2D.from_array(self.array[Point3DIndex.XY], copy=False)

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
