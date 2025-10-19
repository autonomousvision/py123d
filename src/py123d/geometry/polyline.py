from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import shapely.creation as geom_creation
import shapely.geometry as geom
from scipy.interpolate import interp1d

from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry.geometry_index import Point2DIndex, Point3DIndex, StateSE2Index
from py123d.geometry.point import Point2D, Point3D
from py123d.geometry.se import StateSE2
from py123d.geometry.utils.constants import DEFAULT_Z
from py123d.geometry.utils.polyline_utils import get_linestring_yaws, get_path_progress
from py123d.geometry.utils.rotation_utils import normalize_angle

# TODO: Implement PolylineSE3
# TODO: Benchmark interpolation performance and reconsider reliance on LineString


@dataclass
class Polyline2D(ArrayMixin):
    """Represents a interpolatable 2D polyline."""

    linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline2D:
        """Creates a Polyline2D from a Shapely LineString. If the LineString has Z-coordinates, they are ignored.

        :param linestring: A Shapely LineString object.
        :return: A Polyline2D instance.
        """
        if linestring.has_z:
            linestring_ = geom_creation.linestrings(*linestring.xy)
        else:
            linestring_ = linestring
        return Polyline2D(linestring_)

    @classmethod
    def from_array(cls, polyline_array: npt.NDArray[np.float32]) -> Polyline2D:
        """Creates a Polyline2D from a numpy array.

        :param polyline_array: A numpy array of shape (N, 2) or (N, 3), e.g. indexed by \
            :class:`~py123d.geometry.Point2DIndex` or :class:`~py123d.geometry.Point3DIndex`.
        :raises ValueError: If the input array is not of the expected shape.
        :return: A Polyline2D instance.
        """
        assert polyline_array.ndim == 2
        linestring: Optional[geom.LineString] = None
        if polyline_array.shape[-1] == len(Point2DIndex):
            linestring = geom_creation.linestrings(polyline_array)
        elif polyline_array.shape[-1] == len(Point3DIndex):
            linestring = geom_creation.linestrings(polyline_array[:, Point3DIndex.XY])
        else:
            raise ValueError("Array must have shape (N, 2) or (N, 3) for Point2D or Point3D respectively.")
        return Polyline2D(linestring)

    def from_discrete_points(cls, discrete_points: List[Point2D]) -> Polyline2D:
        """Creates a Polyline2D from a list of discrete 2D points.

        :param discrete_points: A list of Point2D instances.
        :return: A Polyline2D instance.
        """
        return Polyline2D.from_array(np.array(discrete_points, dtype=np.float64))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the polyline to a numpy array, indexed by :class:`~py123d.geometry.Point2DIndex`.

        :return: A numpy array of shape (N, 2) representing the polyline.
        """
        x, y = self.linestring.xy
        array = np.zeros((len(x), len(Point2DIndex)), dtype=np.float64)
        array[:, Point2DIndex.X] = x
        array[:, Point2DIndex.Y] = y
        return array

    @property
    def polyline_se2(self) -> PolylineSE2:
        """Converts the 2D polyline to a 2D SE(2) polyline and retrieves the yaw angles.

        :return: A PolylineSE2 instance representing the 2D polyline.
        """
        return PolylineSE2.from_linestring(self.linestring)

    @property
    def length(self) -> float:
        """Returns the length of the polyline.

        :return: The length of the polyline.
        """
        return self.linestring.length

    def interpolate(
        self,
        distances: Union[float, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> Union[Point2D, npt.NDArray[np.float64]]:
        """Interpolates the polyline at the given distances.

        :param distances: The distances at which to interpolate the polyline.
        :return: The interpolated point(s) on the polyline.
        """

        if isinstance(distances, float) or isinstance(distances, int):
            point = self.linestring.interpolate(distances, normalized=normalized)
            return Point2D(point.x, point.y)
        else:
            distances_ = np.asarray(distances, dtype=np.float64)
            points = self.linestring.interpolate(distances, normalized=normalized)
            return np.array([[p.x, p.y] for p in points], dtype=np.float64)

    def project(
        self,
        point: Union[geom.Point, Point2D, StateSE2, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Projects a point onto the polyline and returns the distance along the polyline to the closest point.

        :param point: The point to project onto the polyline.
        :param normalized: Whether to return the normalized distance, defaults to False.
        :return: The distance along the polyline to the closest point.
        """
        if isinstance(point, Point2D) or isinstance(point, StateSE2):
            point_ = point.shapely_point
        elif isinstance(point, geom.Point):
            point_ = point
        else:
            point_ = np.array(point, dtype=np.float64)
        return self.linestring.project(point_, normalized=normalized)


@dataclass
class PolylineSE2(ArrayMixin):
    """Represents a interpolatable SE2 polyline."""

    _array: npt.NDArray[np.float64]
    linestring: Optional[geom.LineString] = None

    _progress: Optional[npt.NDArray[np.float64]] = None
    _interpolator: Optional[interp1d] = None

    def __post_init__(self):
        assert self._array is not None

        if self.linestring is None:
            self.linestring = geom_creation.linestrings(self._array[..., StateSE2Index.XY])

        self._array[:, StateSE2Index.YAW] = np.unwrap(self._array[:, StateSE2Index.YAW], axis=0)
        self._progress = get_path_progress(self._array)
        self._interpolator = interp1d(self._progress, self._array, axis=0, bounds_error=False, fill_value=0.0)

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> PolylineSE2:
        """Creates a PolylineSE2 from a LineString. This requires computing the yaw angles along the path.

        :param linestring: The LineString to convert.
        :return: A PolylineSE2 representing the same path as the LineString.
        """
        points_2d = np.array(linestring.coords, dtype=np.float64)[..., StateSE2Index.XY]
        se2_array = np.zeros((len(points_2d), len(StateSE2Index)), dtype=np.float64)
        se2_array[:, StateSE2Index.XY] = points_2d
        se2_array[:, StateSE2Index.YAW] = get_linestring_yaws(linestring)
        return PolylineSE2(se2_array, linestring)

    @classmethod
    def from_array(cls, polyline_array: npt.NDArray[np.float32]) -> PolylineSE2:
        """Creates a PolylineSE2 from a numpy array.

        :param polyline_array: The input numpy array representing, either indexed by \
            :class:`~py123d.geometry.Point2DIndex` or :class:`~py123d.geometry.StateSE2Index`.
        :raises ValueError: If the input array is not of the expected shape.
        :return: A PolylineSE2 representing the same path as the input array.
        """
        assert polyline_array.ndim == 2
        if polyline_array.shape[-1] == len(Point2DIndex):
            se2_array = np.zeros((len(polyline_array), len(StateSE2Index)), dtype=np.float64)
            se2_array[:, StateSE2Index.XY] = polyline_array
            se2_array[:, StateSE2Index.YAW] = get_linestring_yaws(geom_creation.linestrings(*polyline_array.T))
        elif polyline_array.shape[-1] == len(StateSE2Index):
            se2_array = np.array(polyline_array, dtype=np.float64)
        else:
            raise ValueError("Invalid polyline array shape.")
        return PolylineSE2(se2_array)

    @classmethod
    def from_discrete_se2(cls, discrete_se2: List[StateSE2]) -> PolylineSE2:
        """Creates a PolylineSE2 from a list of discrete SE2 states.

        :param discrete_se2: The list of discrete SE2 states.
        :return: A PolylineSE2 representing the same path as the discrete SE2 states.
        """
        return PolylineSE2.from_array(np.array(discrete_se2, dtype=np.float64))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the polyline to a numpy array, indexed by :class:`~py123d.geometry.StateSE2Index`.

        :return: A numpy array of shape (N, 3) representing the polyline.
        """
        return self._array

    @property
    def length(self) -> float:
        """Returns the length of the polyline.

        :return: The length of the polyline.
        """
        return float(self._progress[-1])

    def interpolate(
        self,
        distances: Union[float, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> Union[StateSE2, npt.NDArray[np.float64]]:
        """Interpolates the polyline at the given distances.

        :param distances: The distances along the polyline to interpolate.
        :param normalized: Whether the distances are normalized (0 to 1), defaults to False
        :return: The interpolated StateSE2 or an array of interpolated states, according to
        """

        distances_ = distances * self.length if normalized else distances
        clipped_distances = np.clip(distances_, 1e-8, self.length)

        interpolated_se2_array = self._interpolator(clipped_distances)
        interpolated_se2_array[..., StateSE2Index.YAW] = normalize_angle(interpolated_se2_array[..., StateSE2Index.YAW])

        if clipped_distances.ndim == 0:
            return StateSE2(*interpolated_se2_array)
        else:
            return interpolated_se2_array

    def project(
        self,
        point: Union[geom.Point, Point2D, StateSE2, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Projects a point onto the polyline and returns the distance along the polyline to the closest point.

        :param point: The point to project onto the polyline.
        :param normalized: Whether to return the normalized distance, defaults to False.
        :return: The distance along the polyline to the closest point.
        """
        if isinstance(point, Point2D) or isinstance(point, StateSE2):
            point_ = point.shapely_point
        elif isinstance(point, geom.Point):
            point_ = point
        else:
            point_ = np.array(point, dtype=np.float64)
        return self.linestring.project(point_, normalized=normalized)


@dataclass
class Polyline3D(ArrayMixin):
    """Represents a interpolatable 3D polyline."""

    linestring: geom.LineString

    @classmethod
    def from_linestring(cls, linestring: geom.LineString) -> Polyline3D:
        """Creates a Polyline3D from a Shapely LineString. If the LineString does not have Z-coordinates, \
            a default Z-value is added.

        :param linestring: The input LineString.
        :return: A Polyline3D instance.
        """
        return (
            Polyline3D(linestring)
            if linestring.has_z
            else Polyline3D(geom_creation.linestrings(*linestring.xy, z=DEFAULT_Z))
        )

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> Polyline3D:
        """Creates a Polyline3D from a numpy array.

        :param array: A numpy array of shape (N, 3) representing 3D points, e.g. indexed by \
            :class:`~py123d.geometry.Point3DIndex`.
        :return: A Polyline3D instance.
        """
        assert array.ndim == 2 and array.shape[1] == len(Point3DIndex), "Array must be 3D with shape (N, 3)"
        linestring = geom_creation.linestrings(*array.T)
        return Polyline3D(linestring)

    @property
    def polyline_2d(self) -> Polyline2D:
        """Converts the 3D polyline to a 2D polyline by dropping the Z-coordinates.

        :return: A Polyline2D instance.
        """
        return Polyline2D(geom_creation.linestrings(*self.linestring.xy))

    @property
    def polyline_se2(self) -> PolylineSE2:
        """Converts the 3D polyline to a 2D SE(2) polyline.

        :return: A PolylineSE2 instance.
        """
        return PolylineSE2.from_linestring(self.linestring)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the 3D polyline to the discrete 3D points.

        :return: A numpy array of shape (N, 3), indexed by :class:`~py123d.geometry.Point3DIndex`.
        """
        return np.array(self.linestring.coords, dtype=np.float64)

    @property
    def length(self) -> float:
        """Returns the length of the 3D polyline.

        :return: The length of the polyline.
        """
        return self.linestring.length

    def interpolate(
        self,
        distances: Union[float, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> Union[Point3D, npt.NDArray[np.float64]]:
        """Interpolates the 3D polyline at the given distances.

        :param distances: A float or numpy array of distances along the polyline.
        :param normalized: Whether to interpret the distances as fractions of the length.
        :return: A Point3D instance or a numpy array of shape (N, 3) representing the interpolated points.
        """

        if isinstance(distances, float) or isinstance(distances, int):
            point = self.linestring.interpolate(distances, normalized=normalized)
            return Point3D(point.x, point.y, point.z)
        else:
            distances = np.asarray(distances, dtype=np.float64)
            points = self.linestring.interpolate(distances, normalized=normalized)
            return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)

    def project(
        self,
        point: Union[geom.Point, Point2D, Point3D, npt.NDArray[np.float64]],
        normalized: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Projects a point onto the 3D polyline and returns the distance along the polyline to the closest point.

        :param point: The point to project.
        :param normalized: Whether to return normalized distances, defaults to False.
        :return: The distance along the polyline to the closest point.
        """
        if isinstance(point, Point2D) or isinstance(point, StateSE2) or isinstance(point, Point3D):
            point_ = point.shapely_point
        elif isinstance(point, geom.Point):
            point_ = point
        else:
            point_ = np.array(point, dtype=np.float64)
        return self.linestring.project(point_, normalized=normalized)


@dataclass
class PolylineSE3:
    # TODO: Implement PolylineSE3 once quaternions are used in StateSE3
    # Interpolating along SE3 states (i.e., 3D position + orientation) is meaningful,
    # but more complex than SE2 due to 3D rotations (quaternions or rotation matrices).
    # Linear interpolation of positions is straightforward, but orientation interpolation
    # should use SLERP (spherical linear interpolation) for quaternions.
    # This is commonly needed in robotics, animation, and path planning.
    pass
