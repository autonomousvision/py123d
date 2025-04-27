from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import shapely.creation as geom_creation
import shapely.geometry as geom

from asim.common.geometry.constants import DEFAULT_Z


@dataclass
class Polyline2D:

    linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline2D:
        return Polyline2D(geom_creation.linestrings(*linestring.xy)) if linestring.has_z else Polyline2D(linestring)

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
    # TODO: implement this class
    pass


class Polyline3D:

    linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: npt.NDArray[np.float32]) -> Polyline2D:
        return (
            Polyline3D(linestring)
            if linestring.has_z
            else Polyline3D(geom_creation.linestrings(*linestring.xy, z=DEFAULT_Z))
        )

    @property
    def polyline_2d(self) -> Polyline2D:
        return Polyline2D.from_linestring(self.linestring)


class PolylineSE3:
    # TODO: implement this class
    pass
