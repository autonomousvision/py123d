from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Union

import numpy as np
import numpy.typing as npt
import shapely

from asim.common.geometry.base import StateSE2, StateSE3
from asim.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index, BoundingBoxSE3Index
from asim.common.geometry.bounding_box.utils import bbse2_array_to_corners_array


@dataclass
class BoundingBoxSE2:

    center: StateSE2
    length: float
    width: float

    @cached_property
    def shapely_polygon(self) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(self.corners_array)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        array = np.zeros(len(BoundingBoxSE2Index), dtype=np.float64)
        array[BoundingBoxSE2Index.X] = self.center.x
        array[BoundingBoxSE2Index.Y] = self.center.y
        array[BoundingBoxSE2Index.YAW] = self.center.yaw
        array[BoundingBoxSE2Index.LENGTH] = self.length
        array[BoundingBoxSE2Index.WIDTH] = self.width
        return array

    @property
    def corners_array(self) -> npt.NDArray[np.float64]:
        return bbse2_array_to_corners_array(self.array)


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
