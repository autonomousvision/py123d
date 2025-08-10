from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import shapely.geometry as geom
import trimesh

from d123.common.geometry.line.polylines import Polyline2D, Polyline3D, PolylineSE2
from d123.dataset.maps.map_datatypes import MapLayer


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

    @property
    @abc.abstractmethod
    def layer(self) -> MapLayer:
        """
        Returns map layer type, e.g. LANE, ROAD_EDGE.
        :return: map layer type
        """


class AbstractSurfaceMapObject(AbstractMapObject):
    """
    Base interface representation of all map objects.
    """

    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        """
        Returns the 2D shapely polygon of the map object.
        :return: shapely polygon
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

    def outline_2d(self) -> Polyline2D:
        return self.outline_3d.polyline_2d


class AbstractLineMapObject(AbstractMapObject):

    @property
    @abc.abstractmethod
    def polyline_3d(self) -> Polyline3D:
        """
        Returns the 3D polyline of the road edge.
        :return: 3D polyline
        """

    @property
    def polyline_2d(self) -> Polyline2D:
        """
        Returns the 2D polyline of the road line.
        :return: 2D polyline
        """
        return self.polyline_3d.polyline_2d

    @property
    def polyline_se2(self) -> PolylineSE2:
        """
        Returns the 2D polyline of the road line in SE(2) coordinates.
        :return: 2D polyline in SE(2)
        """
        return self.polyline_3d.polyline_se2


class AbstractLane(AbstractSurfaceMapObject):
    """Abstract interface for lane objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.LANE

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

    @property
    def boundaries(self) -> Tuple[Polyline3D, Polyline3D]:
        """
        Property of left and right boundary.
        :return: returns tuple of left and right boundary polylines
        """
        return self.left_boundary, self.right_boundary


class AbstractLaneGroup(AbstractSurfaceMapObject):
    """Abstract interface lane groups (nearby lanes going in the same direction)."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.LANE_GROUP

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
    def layer(self) -> MapLayer:
        return MapLayer.INTERSECTION

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
    def layer(self) -> MapLayer:
        return MapLayer.CROSSWALK


class AbstractWalkway(AbstractSurfaceMapObject):
    """Abstract interface for walkway objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.WALKWAY


class AbstractCarpark(AbstractSurfaceMapObject):
    """Abstract interface for carpark objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.CARPARK


class AbstractGenericDrivable(AbstractSurfaceMapObject):
    """Abstract interface for generic drivable objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.GENERIC_DRIVABLE


class AbstractStopLine(AbstractSurfaceMapObject):
    """Abstract interface for stop line objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.STOP_LINE


class AbstractRoadEdge(AbstractLineMapObject):
    """Abstract interface for road edge objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.ROAD_EDGE

    @property
    @abc.abstractmethod
    def polyline_3d(self) -> Polyline3D:
        """
        Returns the 3D polyline of the road edge.
        :return: 3D polyline
        """
        raise NotImplementedError


class AbstractRoadLine(AbstractLineMapObject):
    """Abstract interface for road line objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.ROAD_LINE
