from __future__ import annotations

from typing import Dict, Union

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom

from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr
from py123d.geometry.geometry_index import BoundingBoxSE2Index, BoundingBoxSE3Index, Corners2DIndex, Corners3DIndex
from py123d.geometry.point import Point2D, Point3D
from py123d.geometry.pose import PoseSE2, PoseSE3
from py123d.geometry.utils.bounding_box_utils import bbse2_array_to_corners_array, bbse3_array_to_corners_array


class BoundingBoxSE2(ArrayMixin):
    """
    Rotated bounding box in 2D defined by a center :class:`~py123d.geometry.PoseSE2`, length and width.

    Example:
        >>> from py123d.geometry import PoseSE2, BoundingBoxSE2
        >>> bbox = BoundingBoxSE2(center_se2=PoseSE2(1.0, 2.0, 0.5), length=4.0, width=2.0)
        >>> bbox.array
        array([1. , 2. , 0.5, 4. , 2. ])
        >>> bbox.corners_array.shape
        (4, 2)
        >>> bbox.shapely_polygon.area
        8.0
    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, center_se2: PoseSE2, length: float, width: float):
        """Initialize :class:`BoundingBoxSE2` with :class:`~py123d.geometry.PoseSE2` center, length and width.

        :param center_se2: Center of the bounding box as a :class:`~py123d.geometry.PoseSE2` instance.
        :param length: Length of the bounding box along the x-axis in the local frame.
        :param width: Width of the bounding box along the y-axis in the local frame.
        """
        array = np.zeros(len(BoundingBoxSE2Index), dtype=np.float64)
        array[BoundingBoxSE2Index.SE2] = center_se2.array
        array[BoundingBoxSE2Index.LENGTH] = length
        array[BoundingBoxSE2Index.WIDTH] = width
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> BoundingBoxSE2:
        """Create a :class:`BoundingBoxSE2` from a (5,) numpy array, \
            indexed by :class:`~py123d.geometry.BoundingBoxSE2Index`.

        :param array: A 1D numpy array containing the bounding box parameters.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A :class:`BoundingBoxSE2` instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(BoundingBoxSE2Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def center_se2(self) -> PoseSE2:
        """The center of the bounding box as a :class:`~py123d.geometry.PoseSE2` instance."""
        return PoseSE2.from_array(self._array[BoundingBoxSE2Index.SE2])

    @property
    def length(self) -> float:
        """Length of the bounding box along the x-axis in the local frame."""
        return self._array[BoundingBoxSE2Index.LENGTH]

    @property
    def width(self) -> float:
        """Width of the bounding box along the y-axis in the local frame."""
        return self._array[BoundingBoxSE2Index.WIDTH]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of shape (5,), indexed by :class:`~py123d.geometry.BoundingBoxSE2Index`."""
        return self._array

    @property
    def corners_array(self) -> npt.NDArray[np.float64]:
        """The corner points of the bounding box as a numpy array of shape (4, 2), indexed by \
            :class:`~py123d.geometry.Corners2DIndex` and :class:`~py123d.geometry.Point2DIndex`, respectively.
        """
        return bbse2_array_to_corners_array(self.array)

    @property
    def corners_dict(self) -> Dict[Corners2DIndex, Point2D]:
        """Dictionary of corner points of the bounding box, mapping :class:`~py123d.geometry.Corners2DIndex` to \
            :class:`~py123d.geometry.Point2D` instances.
        """
        corners_array = self.corners_array
        return {index: Point2D.from_array(corners_array[index]) for index in Corners2DIndex}

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """The shapely polygon representation of the bounding box."""
        return geom.Polygon(self.corners_array)

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`BoundingBoxSE2` instance itself."""
        return self

    def __repr__(self) -> str:
        """String representation of :class:`BoundingBoxSE2`."""
        return indexed_array_repr(self, BoundingBoxSE2Index)


class BoundingBoxSE3(ArrayMixin):
    """
    Rotated bounding box in 3D defined by center with quaternion rotation (PoseSE3), length, width and height.

    Example:
        >>> from py123d.geometry import PoseSE3, BoundingBoxSE3
        >>> bbox = BoundingBoxSE3(center_se3=PoseSE3(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0), length=4.0, width=2.0, height=1.5)
        >>> bbox.array
        array([1. , 2. , 3. , 1. , 0. , 0. , 0. , 4. , 2. , 1.5])
        >>> bbox.bounding_box_se2.array
        array([1., 2., 0., 4., 2.])
        >>> bbox.shapely_polygon.area
        8.0
    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, center_se3: PoseSE3, length: float, width: float, height: float):
        """Initialize :class:`BoundingBoxSE3` with :class:`~py123d.geometry.PoseSE3` center, length, width and height.

        :param center_se3: Center of the bounding box as a :class:`~py123d.geometry.PoseSE3` instance.
        :param length: Length of the bounding box along the x-axis in the local frame.
        :param width: Width of the bounding box along the y-axis in the local frame.
        :param height: Height of the bounding box along the z-axis in the local frame.
        """
        array = np.zeros(len(BoundingBoxSE3Index), dtype=np.float64)
        array[BoundingBoxSE3Index.SE3] = center_se3.array
        array[BoundingBoxSE3Index.LENGTH] = length
        array[BoundingBoxSE3Index.WIDTH] = width
        array[BoundingBoxSE3Index.HEIGHT] = height
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> BoundingBoxSE3:
        """Create a :class:`BoundingBoxSE3` from a (10,) numpy array, \
            indexed by :class:`~py123d.geometry.BoundingBoxSE3Index`.

        :param array: A (10,) numpy array containing the bounding box parameters.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A BoundingBoxSE3 instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(BoundingBoxSE3Index)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def center_se3(self) -> PoseSE3:
        """The center of the bounding box as a :class:`~py123d.geometry.PoseSE3` instance."""
        return PoseSE3.from_array(self._array[BoundingBoxSE3Index.SE3])

    @property
    def center_se2(self) -> PoseSE2:
        """The center of the bounding box as a :class:`~py123d.geometry.PoseSE2` instance."""
        return self.center_se3.pose_se2

    @property
    def length(self) -> float:
        """The length of the bounding box along the x-axis in the local frame."""
        return self._array[BoundingBoxSE3Index.LENGTH]

    @property
    def width(self) -> float:
        """The width of the bounding box along the y-axis in the local frame."""
        return self._array[BoundingBoxSE3Index.WIDTH]

    @property
    def height(self) -> float:
        """The height of the bounding box along the z-axis in the local frame."""
        return self._array[BoundingBoxSE3Index.HEIGHT]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array representation of shape (10,), indexed by :class:`~py123d.geometry.BoundingBoxSE3Index`."""
        return self._array

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The SE2 projection :class:`~py123d.geometry.BoundingBoxSE2` of the bounding box."""
        return BoundingBoxSE2(
            center_se2=self.center_se2,
            length=self.length,
            width=self.width,
        )

    @property
    def corners_array(self) -> npt.NDArray[np.float64]:
        """The corner points of the bounding box as a numpy array of shape (8, 3), indexed by \
            :class:`~py123d.geometry.Corners3DIndex` and :class:`~py123d.geometry.Point3DIndex`, respectively.
        """
        return bbse3_array_to_corners_array(self.array)

    @property
    def corners_dict(self) -> Dict[Corners3DIndex, Point3D]:
        """Dictionary of corner points of the bounding box, mapping :class:`~py123d.geometry.Corners3DIndex` to \
            :class:`~py123d.geometry.Point3D` instances.
        """
        corners_array = self.corners_array
        return {index: Point3D.from_array(corners_array[index]) for index in Corners3DIndex}

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """The shapely polygon representation of the SE2 projection of the bounding box."""
        return self.bounding_box_se2.shapely_polygon

    def __repr__(self) -> str:
        """String representation of :class:`BoundingBoxSE3`."""
        return indexed_array_repr(self, BoundingBoxSE3Index)


BoundingBox = Union[BoundingBoxSE2, BoundingBoxSE3]
