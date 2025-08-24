from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import shapely.creation as geom_creation
import shapely.geometry as geom
from scipy.interpolate import interp1d

from d123.geometry.point import Point2D, Point2DIndex, Point3D, Point3DIndex
from d123.geometry.se import StateSE2, StateSE2Index
from d123.geometry.utils.constants import DEFAULT_Z
from d123.geometry.utils.polyline_utils import get_linestring_yaws, get_path_progress
from d123.geometry.utils.rotation_utils import normalize_angle

# TODO: Implement PolylineSE3
# TODO: Benchmark interpolation performance and reconsider reliance on LineString


@dataclass
class Polyline2D:

    linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline2D:
        if linestring.has_z:
            linestring_ = geom_creation.linestrings(*linestring.xy)
        else:
            linestring_ = linestring
        return Polyline2D(linestring_)

    @classmethod
    def from_array(cls, polyline_array: npt.NDArray[np.float32]) -> Polyline2D:
        assert polyline_array.ndim == 2
        linestring: Optional[geom.LineString] = None
        if polyline_array.shape[-1] == len(Point2DIndex):
            linestring = geom_creation.linestrings(polyline_array)
        elif polyline_array.shape[-1] == len(Point3DIndex):
            linestring = geom_creation.linestrings(polyline_array[:, Point3DIndex.XY])
        else:
            raise ValueError("Array must have shape (N, 2) or (N, 3) for Point2D or Point3D respectively.")
        return Polyline2D(linestring)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return np.array(self.linestring.coords, dtype=np.float64)

    @property
    def polyline_se2(self) -> Polyline3D:
        return PolylineSE2.from_linestring(self.linestring)

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

    def project(self, point: Union[Point2D, npt.NDArray[np.float64]]) -> Union[Point2D, npt.NDArray[np.float64]]:
        if isinstance(point, Point2D):
            point_ = point.array
        else:
            point_ = np.array(point, dtype=np.float64)
        return self.linestring.project(point_)


@dataclass
class PolylineSE2:

    se2_array: npt.NDArray[np.float64]
    linestring: Optional[geom.LineString] = None

    _progress: Optional[npt.NDArray[np.float64]] = None
    _interpolator: Optional[interp1d] = None

    def __post_init__(self):
        assert self.se2_array is not None

        if self.linestring is None:
            self.linestring = geom_creation.linestrings(self.se2_array[..., StateSE2Index.XY])

        self.se2_array[:, StateSE2Index.YAW] = np.unwrap(self.se2_array[:, StateSE2Index.YAW], axis=0)
        self._progress = get_path_progress(self.se2_array)
        self._interpolator = interp1d(self._progress, self.se2_array, axis=0, bounds_error=False, fill_value=0.0)

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> PolylineSE2:
        points_2d = np.array(linestring.coords, dtype=np.float64)[..., StateSE2Index.XY]
        se2_array = np.zeros((len(points_2d), len(StateSE2Index)), dtype=np.float64)
        se2_array[:, StateSE2Index.XY] = points_2d
        se2_array[:, StateSE2Index.YAW] = get_linestring_yaws(linestring)
        return PolylineSE2(se2_array, linestring)

    @classmethod
    def from_array(cls, polyline_array: npt.NDArray[np.float32]) -> PolylineSE2:
        assert polyline_array.ndim == 2
        if polyline_array.shape[-1] == len(Point2DIndex):
            se2_array = np.zeros((len(polyline_array), len(StateSE2Index)), dtype=np.float64)
            se2_array[:, StateSE2Index.XY] = polyline_array
            se2_array[:, StateSE2Index.YAW] = get_linestring_yaws(geom_creation.linestrings(*polyline_array.T))
        elif polyline_array.shape[-1] == len(StateSE2Index):
            se2_array = np.array(polyline_array, dtype=np.float64)
        else:
            raise ValueError
        return PolylineSE2(se2_array)

    @classmethod
    def from_discrete_se2(cls, discrete_se2: List[StateSE2]) -> PolylineSE2:
        return PolylineSE2(np.array([se2.array for se2 in discrete_se2], dtype=np.float64))

    @property
    def length(self) -> float:
        return float(self._progress[-1])

    def interpolate(self, distances: Union[float, npt.NDArray[np.float64]]) -> Union[StateSE2, npt.NDArray[np.float64]]:
        clipped_distances = np.clip(distances, 1e-8, self.length)
        interpolated_se2_array = self._interpolator(clipped_distances)
        interpolated_se2_array[..., StateSE2Index.YAW] = normalize_angle(interpolated_se2_array[..., StateSE2Index.YAW])

        if clipped_distances.ndim == 0:
            return StateSE2(*interpolated_se2_array)
        else:
            return interpolated_se2_array

    def project(
        self, point: Union[geom.Point, Point2D, npt.NDArray[np.float64]]
    ) -> Union[Point2D, npt.NDArray[np.float64]]:
        if isinstance(point, Point2D):
            point_ = geom.Point(point.x, point.y)
        elif isinstance(point, np.ndarray) and point.shape[-1] == 2:
            point_ = geom_creation.points(point)
        elif isinstance(point, geom.Point):
            point_ = point
        else:
            raise ValueError("Point must be a Point2D, geom.Point, or a 2D numpy array.")

        return self.linestring.project(point_)


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

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> Polyline3D:
        assert array.ndim == 2 and array.shape[1] == 3, "Array must be 2D with shape (N, 3)"
        linestring = geom_creation.linestrings(*array.T)
        return Polyline3D(linestring)

    @property
    def polyline_2d(self) -> Polyline2D:
        return Polyline2D(geom_creation.linestrings(*self.linestring.xy))

    @property
    def polyline_se2(self) -> PolylineSE2:
        return PolylineSE2.from_linestring(self.linestring)

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
