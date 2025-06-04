from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt
import shapely.creation as geom_creation
import shapely.geometry as geom

from asim.common.geometry.base import Point2D, Point3D
from asim.common.geometry.constants import DEFAULT_Z


@dataclass
class Polyline2D:

    linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline2D:
        return Polyline2D(linestring)

    @classmethod
    def from_array(cls, polyline_array: npt.NDArray[np.float32]) -> Polyline2D:
        raise NotImplementedError

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return np.array(self.linestring.coords, dtype=np.float64)

    @property
    def length(self) -> float:
        return self.linestring.length

    def interpolate(self, distances: Union[float, npt.NDArray[np.float64]]) -> Union[Point2D, npt.NDArray[np.float64]]:
        if isinstance(distances, float) or isinstance(distances, int):
            point = self.linestring.interpolate(distances)
            return Point2D(point.x, point.y)
        else:
            distances = np.asarray(distances, dtype=np.float64)
            points = self.linestring.interpolate(distances)
            return np.array([[p.x, p.y] for p in points], dtype=np.float64)


@dataclass
class PolylineSE2:
    # TODO: implement this class
    pass


@dataclass
class Polyline3D:

    linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline3D:
        return (
            Polyline3D(linestring)
            if linestring.has_z
            else Polyline3D(geom_creation.linestrings(*linestring.xy, z=DEFAULT_Z))
        )

    @property
    def polyline_2d(self) -> Polyline2D:
        return Polyline2D.from_linestring(self.linestring)

    @property
    def array(self) -> Polyline2D:
        return np.array(self.linestring.coords, dtype=np.float64)

    @property
    def length(self) -> float:
        return self.linestring.length

    def interpolate(self, distances: Union[float, npt.NDArray[np.float64]]) -> Union[Point3D, npt.NDArray[np.float64]]:
        if isinstance(distances, float) or isinstance(distances, int):
            point = self.linestring.interpolate(distances)
            return Point3D(point.x, point.y, point.z)
        else:
            distances = np.asarray(distances, dtype=np.float64)
            points = self.linestring.interpolate(distances)
            return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)


@dataclass
class PolylineSE3:
    # TODO: implement this class
    pass
