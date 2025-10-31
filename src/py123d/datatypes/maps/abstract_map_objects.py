from __future__ import annotations

import abc
from typing import List, Optional, Tuple, Union
from typing_extensions import TypeAlias


import shapely.geometry as geom
import trimesh

from py123d.datatypes.maps.map_datatypes import MapLayer, RoadEdgeType, RoadLineType
from py123d.geometry import Polyline2D, Polyline3D, PolylineSE2

# TODO: Refactor and just use int
# type MapObjectIDType = Union[str, int] for Python >= 3.12
MapObjectIDType: TypeAlias = Union[str, int]


class AbstractMapObject(abc.ABC):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_id: MapObjectIDType):
        """
        Constructor of the base map object type.
        :param object_id: unique identifier of the map object.
        """
        self.object_id: MapObjectIDType = object_id

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
    def outline(self) -> Union[Polyline2D, Polyline3D]:
        """
        Returns the 2D or 3D outline of the map surface, if available.
        :return: 2D or 3D polyline
        """

    @property
    @abc.abstractmethod
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """
        Returns a triangle mesh of the map surface.
        :return: Trimesh
        """

    @property
    def outline_3d(self) -> Polyline3D:
        """Returns the 3D outline of the map surface, or converts 2D to 3D if necessary.

        :return: 3D polyline
        """
        if isinstance(self.outline, Polyline3D):
            return self.outline
        # Converts 2D polyline to 3D by adding a default (zero) z-coordinate
        return Polyline3D.from_linestring(self.outline.linestring)

    @property
    def outline_2d(self) -> Polyline2D:
        """Returns the 2D outline of the map surface, or converts 3D to 2D if necessary.

        :return: 2D polyline
        """
        if isinstance(self.outline, Polyline2D):
            return self.outline
        # Converts 3D polyline to 2D by dropping the z-coordinate
        return self.outline.polyline_2d


class AbstractLineMapObject(AbstractMapObject):

    @property
    @abc.abstractmethod
    def polyline(self) -> Union[Polyline2D, Polyline3D]:
        """
        Returns the polyline of the road edge, either 2D or 3D.
        :return: polyline
        """

    @property
    def polyline_3d(self) -> Polyline3D:
        """
        Returns the 3D polyline of the road edge.
        :return: 3D polyline
        """
        if isinstance(self.polyline, Polyline3D):
            return self.polyline
        # Converts 2D polyline to 3D by adding a default (zero) z-coordinate
        return Polyline3D.from_linestring(self.polyline.linestring)

    @property
    def polyline_2d(self) -> Polyline2D:
        """
        Returns the 2D polyline of the road line.
        :return: 2D polyline
        """
        if isinstance(self.polyline, Polyline2D):
            return self.polyline
        # Converts 3D polyline to 2D by dropping the z-coordinate
        return self.polyline.polyline_2d

    @property
    def polyline_se2(self) -> PolylineSE2:
        """
        Returns the 2D polyline of the road line in SE(2) coordinates.
        :return: 2D polyline in SE(2)
        """
        return self.polyline_2d.polyline_se2

    @property
    def shapely_linestring(self) -> geom.LineString:
        """
        Returns the shapely linestring of the line, either 2D or 3D.
        :return: shapely linestring
        """
        return self.polyline.linestring


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
    def successor_ids(self) -> List[MapObjectIDType]:
        """
        Property of succeeding lane object ids (front).
        :return: list of lane ids
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
    def predecessor_ids(self) -> List[MapObjectIDType]:
        """
        Property of preceding lane object ids (behind).
        :return: list of lane ids
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
    def left_lane_id(self) -> Optional[MapObjectIDType]:
        """
        Property of left lane id of lane.
        :return: returns left lane id or none, if no left lane
        """

    @property
    @abc.abstractmethod
    def left_lane(self) -> Optional[AbstractLane]:
        """
        Property of left lane of lane.
        :return: returns left lane or none, if no left lane
        """

    @property
    @abc.abstractmethod
    def right_lane_id(self) -> Optional[MapObjectIDType]:
        """
        Property of right lane id of lane.
        :return: returns right lane id or none, if no right lane
        """

    @property
    @abc.abstractmethod
    def right_lane(self) -> Optional[AbstractLane]:
        """
        Property of right lane of lane.
        :return: returns right lane or none, if no right lane
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
    def lane_group_id(self) -> AbstractLaneGroup:
        """
        Property of lane group id of lane.
        :return: returns lane group id
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
    def successor_ids(self) -> List[MapObjectIDType]:
        """
        Property of succeeding lane object ids (front).
        :return: list of lane group ids
        """

    @property
    @abc.abstractmethod
    def successors(self) -> List[AbstractLaneGroup]:
        """
        Property of succeeding lane group objects (front).
        :return: list of lane group class
        """

    @property
    @abc.abstractmethod
    def predecessor_ids(self) -> List[MapObjectIDType]:
        """
        Property of preceding lane object ids (behind).
        :return: list of lane group ids
        """

    @property
    @abc.abstractmethod
    def predecessors(self) -> List[AbstractLaneGroup]:
        """
        Property of preceding lane group objects (behind).
        :return: list of lane group class
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
    def lane_ids(self) -> List[MapObjectIDType]:
        """
        Property of interior lane ids of a lane group.
        :return: returns list of lane ids
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
    def intersection_id(self) -> Optional[MapObjectIDType]:
        """
        Property of intersection id of a lane group.
        :return: returns intersection id or none, if lane group not in intersection
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
    def lane_group_ids(self) -> List[MapObjectIDType]:
        """
        Property of lane group ids of intersection.
        :return: returns list of lane group ids
        """

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
    def road_edge_type(self) -> RoadEdgeType:
        """
        Returns the road edge type.
        :return: RoadEdgeType
        """


class AbstractRoadLine(AbstractLineMapObject):
    """Abstract interface for road line objects."""

    @property
    def layer(self) -> MapLayer:
        return MapLayer.ROAD_LINE

    @property
    @abc.abstractmethod
    def road_line_type(self) -> RoadLineType:
        """
        Returns the road line type.
        :return: RoadLineType
        """
