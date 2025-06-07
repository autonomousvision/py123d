from __future__ import annotations

import abc
from typing import List, Optional

import shapely.geometry as geom
import trimesh

from asim.common.geometry.line.polylines import Polyline3D
from asim.dataset.maps.map_datatypes import MapSurfaceType


class AbstractMapObject(abc.ABC):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_id: str):
        """
        Constructor of the base map object type.
        :param object_id: unique identifier of the map object.
        """
        self.id = str(object_id)


class AbstractSurfaceMapObject(AbstractMapObject):
    """
    Base interface representation of all map objects.
    """

    # pass
    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        """
        Returns the 2D shapely polygon of the map object.
        :return: shapely polygon
        """

    @property
    @abc.abstractmethod
    def surface_type(self) -> MapSurfaceType:
        """
        Returns map surface type, e.g. LANE.
        :return: map surface type
        """

    @property
    @abc.abstractmethod
    def outline_3d(self) -> Polyline3D:
        """
        Returns the 3D outline of the map object.
        :return: 3D polyline
        """

    @property
    @abc.abstractmethod
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """
        Returns a triangle mesh of the map surface.
        :return: Trimesh
        """

    # @property
    # @abc.abstractmethod
    # def outline_2d(self) -> Polyline2D:
    #     pass


class AbstractLane(AbstractSurfaceMapObject):
    """Abstract interface for lane objects."""

    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.LANE

    @property
    @abc.abstractmethod
    def speed_limit_mps(self) -> Optional[float]:
        """
        Property of lanes speed limit in m/s, if available.
        :return: float or none
        """

    @property
    @abc.abstractmethod
    def successors(self) -> List[AbstractLane]:
        """
        Property of succeeding lane objects (front).
        :return: list of lane class
        """

    @property
    @abc.abstractmethod
    def predecessors(self) -> List[AbstractLane]:
        """
        Property of preceding lane objects (behind).
        :return: list of lane class
        """

    @property
    @abc.abstractmethod
    def left_boundary(self) -> Polyline3D:
        """
        Property of left boundary of lane.
        :return: returns 3D polyline
        """

    @property
    @abc.abstractmethod
    def right_boundary(self) -> Polyline3D:
        """
        Property of right boundary of lane.
        :return: returns 3D polyline
        """

    @property
    @abc.abstractmethod
    def centerline(self) -> Polyline3D:
        """
        Property of centerline of lane.
        :return: returns 3D polyline
        """

    @property
    @abc.abstractmethod
    def lane_group(self) -> AbstractLaneGroup:
        """
        Property of lane group of lane.
        :return: returns lane group
        """


class AbstractLaneGroup(AbstractSurfaceMapObject):
    """Abstract interface lane groups (nearby lanes going in the same direction)."""

    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.LANE_GROUP

    @property
    @abc.abstractmethod
    def successors(self) -> List[AbstractLaneGroup]:
        """
        Property of succeeding lane objects (front).
        :return: list of lane class
        """

    @property
    @abc.abstractmethod
    def predecessors(self) -> List[AbstractLaneGroup]:
        """
        Property of preceding lane objects (behind).
        :return: list of lane class
        """

    @property
    @abc.abstractmethod
    def left_boundary(self) -> Polyline3D:
        """
        Property of left boundary of lane group.
        :return: returns 3D polyline
        """

    @property
    @abc.abstractmethod
    def right_boundary(self) -> Polyline3D:
        """
        Property of right boundary of lane group.
        :return: returns 3D polyline
        """

    @property
    @abc.abstractmethod
    def lanes(self) -> List[AbstractLane]:
        """
        Property of interior lanes of a lane group.
        :return: returns list of lanes
        """

    @property
    @abc.abstractmethod
    def intersection(self) -> Optional[AbstractIntersection]:
        """
        Property of intersection of a lane group.
        :return: returns intersection or none, if lane group not in intersection
        """


class AbstractIntersection(AbstractSurfaceMapObject):
    """Abstract interface for intersection objects."""

    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.INTERSECTION

    @property
    @abc.abstractmethod
    def lane_groups(self) -> List[AbstractLaneGroup]:
        """
        Property of lane groups of intersection.
        :return: returns list of lane groups
        """


class AbstractCrosswalk(AbstractSurfaceMapObject):
    """Abstract interface for crosswalk objects."""

    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.CROSSWALK


class AbstractWalkway(AbstractSurfaceMapObject):
    """Abstract interface for walkway objects."""

    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.WALKWAY


class AbstractCarpark(AbstractSurfaceMapObject):
    """Abstract interface for carpark objects."""

    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.CARPARK


class AbstractGenericDrivable(AbstractSurfaceMapObject):
    """Abstract interface for generic drivable objects."""

    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.GENERIC_DRIVABLE
