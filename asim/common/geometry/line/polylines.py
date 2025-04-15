from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString


@dataclass
class Polyline2D:

    linestring: LineString

    @classmethod
    def from_linestring(cls, linestring: npt.NDArray[np.float32]) -> Polyline2D:
        raise Polyline2D(linestring)

    @classmethod
    def from_array(cls, polyline_array: npt.NDArray[np.float32]) -> Polyline2D:
        raise NotImplementedError

    @property
    def array(self):
        raise NotImplementedError

    @property
    def linestring(self):
        raise NotImplementedError

    def interpolate(self, distances: float):
        raise NotImplementedError


class PolylineSE2:
    pass


class Polyline3D:
    pass


class PolylineSE3:
    pass
