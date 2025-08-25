from __future__ import annotations

from ast import Dict
from functools import cached_property
from typing import Union

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from d123.common.utils.mixin import ArrayMixin
from d123.geometry.geometry_index import BoundingBoxSE2Index, BoundingBoxSE3Index, Corners2DIndex, Corners3DIndex
from d123.geometry.point import Point2D, Point3D
from d123.geometry.se import StateSE2, StateSE3
from d123.geometry.utils.bounding_box_utils import bbse2_array_to_corners_array, bbse3_array_to_corners_array


class BoundingBoxSE2(ArrayMixin):
    """
    Rotated bounding box in 2D defined by center (StateSE2), length and width.

    Example:
        >>> from d123.geometry import StateSE2
        >>> bbox = BoundingBoxSE2(center=StateSE2(1.0, 2.0, 0.5), length=4.0, width=2.0)
        >>> bbox.array
        array([1. , 2. , 0.5, 4. , 2. ])
        >>> bbox.corners_array.shape
        (4, 2)
        >>> bbox.shapely_polygon.area
        8.0
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, center: StateSE2, length: float, width: float):
        """Initialize BoundingBoxSE2 with center (StateSE2), length and width.

        :param center: Center of the bounding box as a StateSE2 instance.
        :param length: Length of the bounding box along the x-axis in the local frame.
        :param width: Width of the bounding box along the y-axis in the local frame.
        """
        array = np.zeros(len(BoundingBoxSE2Index), dtype=np.float64)
        array[BoundingBoxSE2Index.SE2] = center.array
        array[BoundingBoxSE2Index.LENGTH] = length
        array[BoundingBoxSE2Index.WIDTH] = width
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> BoundingBoxSE2:
        """Create a BoundingBoxSE2 from a numpy array.

        :param array: A 1D numpy array containing the bounding box parameters, indexed by \
            :class:`~d123.geometry.BoundingBoxSE2Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A BoundingBoxSE2 instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(BoundingBoxSE2Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def center(self) -> StateSE2:
        """The center of the bounding box as a StateSE2 instance.

        :return: The center of the bounding box as a StateSE2 instance.
        """
        return StateSE2.from_array(self._array[BoundingBoxSE2Index.SE2])

    @property
    def center_se2(self) -> StateSE2:
        """The center of the bounding box as a StateSE2 instance.

        :return: The center of the bounding box as a StateSE2 instance.
        """
        return self.center

    @property
    def length(self) -> float:
        """The length of the bounding box along the x-axis in the local frame.

        :return: The length of the bounding box.
        """
        return self._array[BoundingBoxSE2Index.LENGTH]

    @property
    def width(self) -> float:
        """The width of the bounding box along the y-axis in the local frame.

        :return: The width of the bounding box.
        """
        return self._array[BoundingBoxSE2Index.WIDTH]

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the BoundingBoxSE2 instance to a numpy array, indexed by :class:`~d123.geometry.BoundingBoxSE2Index`.

        :return: A numpy array of shape (5,) containing the bounding box parameters [x, y, yaw, length, width].
        """
        return self._array

    @cached_property
    def shapely_polygon(self) -> geom.Polygon:
        """Return a Shapely polygon representation of the bounding box.

        :return: A Shapely polygon representing the bounding box.
        """
        return geom.Polygon(self.corners_array)

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """Returns bounding box itself for polymorphism.

        :return: A BoundingBoxSE2 instance representing the 2D bounding box.
        """
        return self

    @cached_property
    def corners_array(self) -> npt.NDArray[np.float64]:
        """Returns the corner points of the bounding box as a numpy array.

        :return: A numpy array of shape (4, 2) containing the corner points of the bounding box, \
            indexed by :class:`~d123.geometry.Corners2DIndex` and :class:`~d123.geometry.Point2DIndex`.
        """
        return bbse2_array_to_corners_array(self.array)

    @property
    def corners_dict(self) -> Dict[Corners2DIndex, Point2D]:
        """Returns the corner points of the bounding box as a dictionary.

        :return: A dictionary mapping :class:`~d123.geometry.Corners2DIndex` to :class:`~d123.geometry.Point2D` instances.
        """
        corners_array = self.corners_array
        return {index: Point2D.from_array(corners_array[index]) for index in Corners2DIndex}


class BoundingBoxSE3(ArrayMixin):
    """
    Rotated bounding box in 3D defined by center (StateSE3), length, width and height.

    Example:
        >>> from d123.geometry import StateSE3
        >>> bbox = BoundingBoxSE3(center=StateSE3(1.0, 2.0, 3.0, 0.1, 0.2, 0.3), length=4.0, width=2.0, height=1.5)
        >>> bbox.array
        array([1. , 2. , 3. , 0.1, 0.2, 0.3, 4. , 2. , 1.5])
        >>> bbox.bounding_box_se2.array
        array([1. , 2. , 0.3, 4. , 2. ])
        >>> bbox.shapely_polygon.area
        8.0
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, center: StateSE3, length: float, width: float, height: float):
        """Initialize BoundingBoxSE3 with center (StateSE3), length, width and height.

        :param center: Center of the bounding box as a StateSE3 instance.
        :param length: Length of the bounding box along the x-axis in the local frame.
        :param width: Width of the bounding box along the y-axis in the local frame.
        :param height: Height of the bounding box along the z-axis in the local frame.
        """
        array = np.zeros(len(BoundingBoxSE3Index), dtype=np.float64)
        array[BoundingBoxSE3Index.STATE_SE3] = center.array
        array[BoundingBoxSE3Index.LENGTH] = length
        array[BoundingBoxSE3Index.WIDTH] = width
        array[BoundingBoxSE3Index.HEIGHT] = height
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> BoundingBoxSE3:
        """Create a BoundingBoxSE3 from a numpy array.

        :param array: A 1D numpy array containing the bounding box parameters, indexed by \
            :class:`~d123.geometry.BoundingBoxSE3Index`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A BoundingBoxSE3 instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(BoundingBoxSE3Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def center(self) -> StateSE3:
        """The center of the bounding box as a StateSE3 instance.

        :return: The center of the bounding box as a StateSE3 instance.
        """
        return StateSE3.from_array(self._array[BoundingBoxSE3Index.STATE_SE3])

    @property
    def center_se3(self) -> StateSE3:
        """The center of the bounding box as a StateSE3 instance.

        :return: The center of the bounding box as a StateSE3 instance.
        """
        return self.center

    @property
    def length(self) -> float:
        """The length of the bounding box along the x-axis in the local frame.

        :return: The length of the bounding box.
        """
        return self._array[BoundingBoxSE3Index.LENGTH]

    @property
    def width(self) -> float:
        """The width of the bounding box along the y-axis in the local frame.

        :return: The width of the bounding box.
        """
        return self._array[BoundingBoxSE3Index.WIDTH]

    @property
    def height(self) -> float:
        """The height of the bounding box along the z-axis in the local frame.

        :return: The height of the bounding box.
        """
        return self._array[BoundingBoxSE3Index.HEIGHT]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Convert the BoundingBoxSE3 instance to a numpy array.

        :return: A 1D numpy array containing the bounding box parameters, indexed by \
            :class:`~d123.geometry.BoundingBoxSE3Index`.
        """
        return self._array

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """Converts the 3D bounding box to a 2D bounding box by dropping the z, roll and pitch components.

        :return: A BoundingBoxSE2 instance.
        """
        center_se3 = self.center_se3
        return BoundingBoxSE2(
            center=StateSE2(center_se3.x, center_se3.y, center_se3.yaw),
            length=self.length,
            width=self.width,
        )

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """Return a Shapely polygon representation of the 2D projection of the bounding box.

        :return: A shapely polygon representing the 2D bounding box.
        """
        return self.bounding_box_se2.shapely_polygon

    @cached_property
    def corners_array(self) -> npt.NDArray[np.float64]:
        """Returns the corner points of the bounding box as a numpy array, shape (8, 3).

        :return: A numpy array of shape (8, 3) containing the corner points of the bounding box, \
            indexed by :class:`~d123.geometry.Corners3DIndex` and :class:`~d123.geometry.Point3DIndex`.
        """
        return bbse3_array_to_corners_array(self.array)

    @cached_property
    def corners_dict(self) -> Dict[Corners3DIndex, Point3D]:
        """Returns the corner points of the bounding box as a dictionary.

        :return: A dictionary mapping :class:`~d123.geometry.Corners3DIndex` to \
            :class:`~d123.geometry.Point3D` instances.
        """
        corners_array = self.corners_array
        return {index: Point3D.from_array(corners_array[index]) for index in Corners3DIndex}


BoundingBox = Union[BoundingBoxSE2, BoundingBoxSE3]
