from __future__ import annotations

from ast import Dict
from dataclasses import dataclass
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


@dataclass
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

    center: StateSE2
    length: float
    width: float

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> BoundingBoxSE2:
        """Create a BoundingBoxSE2 from a numpy array, index by :class:`~d123.geometry.BoundingBoxSE2Index`.

        :param array: A 1D numpy array containing the bounding box parameters.
        :return: A BoundingBoxSE2 instance.
        """
        assert array.ndim == 1 and array.shape[-1] == len(BoundingBoxSE2Index)
        return BoundingBoxSE2(
            center=StateSE2.from_array(array[BoundingBoxSE2Index.SE2]),
            length=array[BoundingBoxSE2Index.LENGTH],
            width=array[BoundingBoxSE2Index.WIDTH],
        )

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the BoundingBoxSE2 instance to a numpy array, indexed by :class:`~d123.geometry.BoundingBoxSE2Index`.

        :return: A numpy array of shape (5,) containing the bounding box parameters [x, y, yaw, length, width].
        """
        array = np.zeros(len(BoundingBoxSE2Index), dtype=np.float64)
        array[BoundingBoxSE2Index.X] = self.center.x
        array[BoundingBoxSE2Index.Y] = self.center.y
        array[BoundingBoxSE2Index.YAW] = self.center.yaw
        array[BoundingBoxSE2Index.LENGTH] = self.length
        array[BoundingBoxSE2Index.WIDTH] = self.width
        return array

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


@dataclass
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

    center: StateSE3
    length: float
    width: float
    height: float

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> BoundingBoxSE3:
        """Create a BoundingBoxSE3 from a numpy array.

        :param array: A 1D numpy array containing the bounding box parameters, indexed by \
            :class:`~d123.geometry.BoundingBoxSE3Index`.
        :return: A BoundingBoxSE3 instance.
        """
        assert array.ndim == 1 and array.shape[-1] == len(BoundingBoxSE3Index)
        return BoundingBoxSE3(
            center=StateSE3.from_array(array[BoundingBoxSE3Index.STATE_SE3]),
            length=array[BoundingBoxSE3Index.LENGTH],
            width=array[BoundingBoxSE3Index.WIDTH],
            height=array[BoundingBoxSE3Index.HEIGHT],
        )

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        """Convert the BoundingBoxSE3 instance to a numpy array.

        :return: A 1D numpy array containing the bounding box parameters, indexed by \
            :class:`~d123.geometry.BoundingBoxSE3Index`.
        """
        array = np.zeros(len(BoundingBoxSE3Index), dtype=np.float64)
        array[BoundingBoxSE3Index.X] = self.center.x
        array[BoundingBoxSE3Index.Y] = self.center.y
        array[BoundingBoxSE3Index.Z] = self.center.z
        array[BoundingBoxSE3Index.ROLL] = self.center.roll
        array[BoundingBoxSE3Index.PITCH] = self.center.pitch
        array[BoundingBoxSE3Index.YAW] = self.center.yaw
        array[BoundingBoxSE3Index.LENGTH] = self.length
        array[BoundingBoxSE3Index.WIDTH] = self.width
        array[BoundingBoxSE3Index.HEIGHT] = self.height
        return array

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """Converts the 3D bounding box to a 2D bounding box by dropping the z, roll and pitch components.

        :return: A BoundingBoxSE2 instance.
        """
        return BoundingBoxSE2(
            center=StateSE2(self.center.x, self.center.y, self.center.yaw),
            length=self.length,
            width=self.width,
        )

    @property
    def center_se3(self) -> StateSE3:
        """Returns the center of the bounding box as a StateSE3 instance.

        :return: The center of the bounding box as a StateSE3 instance.
        """
        return self.center

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
