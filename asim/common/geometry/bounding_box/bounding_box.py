from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Union

import numpy as np
import numpy.typing as npt
import shapely

from asim.common.geometry.base import Point2D, StateSE2, StateSE3
from asim.common.geometry.tranform_2d import translate_along_yaw
from asim.common.utils.enums import classproperty


class BoundingBoxSE2Index(IntEnum):
    X = 0
    Y = 1
    YAW = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


@dataclass
class BoundingBoxSE2:

    center: StateSE2
    length: float
    width: float

    @property
    def shapely_polygon(self) -> shapely.geometry.Polygon:

        return shapely.geometry.Polygon(
            [
                translate_along_yaw(self.center, Point2D(self.length / 2.0, self.width / 2.0)).point_2d.array,
                translate_along_yaw(self.center, Point2D(self.length / 2.0, -self.width / 2.0)).point_2d.array,
                translate_along_yaw(self.center, Point2D(-self.length / 2.0, -self.width / 2.0)).point_2d.array,
                translate_along_yaw(self.center, Point2D(-self.length / 2.0, self.width / 2.0)).point_2d.array,
            ]
        )


class BoundingBoxSE3Index(IntEnum):
    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    LENGTH = 6
    WIDTH = 7
    HEIGHT = 8

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def STATE_SE3(cls) -> slice:
        return slice(cls.X, cls.YAW + 1)

    @classproperty
    def ROTATION_XYZ(cls) -> slice:
        return slice(cls.ROLL, cls.YAW + 1)


@dataclass
class BoundingBoxSE3:

    center: StateSE3
    length: float
    width: float
    height: float

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> BoundingBoxSE3:
        return cls(
            center=StateSE3.from_array(array[BoundingBoxSE3Index.STATE_SE3]),
            length=array[BoundingBoxSE3Index.LENGTH],
            width=array[BoundingBoxSE3Index.WIDTH],
            height=array[BoundingBoxSE3Index.HEIGHT],
        )

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        return BoundingBoxSE2(
            center=StateSE2(self.center.x, self.center.y, self.center.yaw),
            length=self.length,
            width=self.width,
        )

    @property
    def array(self) -> npt.NDArray[np.float64]:
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
    def shapely_polygon(self) -> shapely.geometry.Polygon:
        return self.bounding_box_se2.shapely_polygon


BoundingBox = Union[BoundingBoxSE2, BoundingBoxSE3]
