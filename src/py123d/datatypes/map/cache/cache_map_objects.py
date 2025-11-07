from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import shapely.geometry as geom
import trimesh

from py123d.datatypes.map.abstract_map_objects import (
    AbstractCarpark,
    AbstractCrosswalk,
    AbstractGenericDrivable,
    AbstractIntersection,
    AbstractLane,
    AbstractLaneGroup,
    AbstractLineMapObject,
    AbstractRoadEdge,
    AbstractRoadLine,
    AbstractSurfaceMapObject,
    AbstractWalkway,
    MapObjectIDType,
)
from py123d.datatypes.map.map_datatypes import MapLayer, RoadEdgeType, RoadLineType
from py123d.geometry import Polyline3D
from py123d.geometry.polyline import Polyline2D


class CacheSurfaceObject(AbstractSurfaceMapObject):
    """
    Base interface representation of all map objects.
    """

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ) -> None:
        super().__init__(object_id)

        assert outline is not None or geometry is not None, "Either outline or geometry must be provided."

        if outline is None:
            outline = Polyline3D.from_linestring(geometry.exterior)

        if geometry is None:
            geometry = geom.Polygon(outline.array[:, :2])

        self._outline = outline
        self._geometry = geometry

    outline = property(lambda self: self._outline)

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """Inherited, see superclass."""
        return self._geometry

    @property
    def outline_3d(self) -> Polyline3D:
        """Inherited, see superclass."""
        if isinstance(self.outline, Polyline3D):
            return self.outline
        # Converts 2D polyline to 3D by adding a default (zero) z-coordinate
        return Polyline3D.from_linestring(self.outline.linestring)

    @property
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """Inherited, see superclass."""
        raise NotImplementedError


class CacheLineObject(AbstractLineMapObject):

    def __init__(self, object_id: MapObjectIDType, polyline: Union[Polyline2D, Polyline3D]) -> None:
        """
        Constructor of the base line map object type.
        :param object_id: unique identifier of a line map object.
        """
        super().__init__(object_id)
        self._polyline = polyline

    polyline = property(lambda self: self._polyline)


class CacheLane(CacheSurfaceObject, AbstractLane):

    def __init__(
        self,
        object_id: MapObjectIDType,
        lane_group_id: MapObjectIDType,
        left_boundary: Polyline3D,
        right_boundary: Polyline3D,
        centerline: Polyline3D,
        left_lane_id: Optional[MapObjectIDType] = None,
        right_lane_id: Optional[MapObjectIDType] = None,
        predecessor_ids: List[MapObjectIDType] = [],
        successor_ids: List[MapObjectIDType] = [],
        speed_limit_mps: Optional[float] = None,
        outline: Optional[Polyline3D] = None,
        geometry: Optional[geom.Polygon] = None,
    ) -> None:

        if outline is None:
            outline_array = np.vstack(
                (
                    left_boundary.array,
                    right_boundary.array[::-1],
                    left_boundary.array[0],
                )
            )
            outline = Polyline3D.from_linestring(geom.LineString(outline_array))

        super().__init__(object_id, outline, geometry)

        self._lane_group_id = lane_group_id
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._centerline = centerline
        self._left_lane_id = left_lane_id
        self._right_lane_id = right_lane_id
        self._predecessor_ids = predecessor_ids
        self._successor_ids = successor_ids
        self._speed_limit_mps = speed_limit_mps

    lane_group_id = property(lambda self: self._lane_group_id)
    left_boundary = property(lambda self: self._left_boundary)
    right_boundary = property(lambda self: self._right_boundary)
    centerline = property(lambda self: self._centerline)
    left_lane_id = property(lambda self: self._left_lane_id)
    right_lane_id = property(lambda self: self._right_lane_id)
    predecessor_ids = property(lambda self: self._predecessor_ids)
    successor_ids = property(lambda self: self._successor_ids)
    speed_limit_mps = property(lambda self: self._speed_limit_mps)

    @property
    def layer(self) -> MapLayer:
        """Inherited, see superclass."""
        return MapLayer.LANE

    @property
    def successors(self) -> List[AbstractLane]:
        """Inherited, see superclass."""
        raise NotImplementedError

    @property
    def predecessors(self) -> List[AbstractLane]:
        """Inherited, see superclass."""
        raise NotImplementedError

    @property
    def left_lane(self) -> Optional[AbstractLane]:
        """Inherited, see superclass."""
        raise NotImplementedError

    @property
    def right_lane(self) -> Optional[AbstractLane]:
        """Inherited, see superclass."""
        raise NotImplementedError

    @property
    def lane_group(self) -> AbstractLaneGroup:
        """Inherited, see superclass."""
        raise NotImplementedError


class CacheLaneGroup(CacheSurfaceObject, AbstractLaneGroup):
    def __init__(
        self,
        object_id: MapObjectIDType,
        lane_ids: List[MapObjectIDType],
        left_boundary: Polyline3D,
        right_boundary: Polyline3D,
        intersection_id: Optional[MapObjectIDType] = None,
        predecessor_ids: List[MapObjectIDType] = [],
        successor_ids: List[MapObjectIDType] = [],
        outline: Optional[Polyline3D] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        if outline is None:
            outline_array = np.vstack(
                (
                    left_boundary.array,
                    right_boundary.array[::-1],
                    left_boundary.array[0],
                )
            )
            outline = Polyline3D.from_linestring(geom.LineString(outline_array))
        super().__init__(object_id, outline, geometry)

        self._lane_ids = lane_ids
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._intersection_id = intersection_id
        self._predecessor_ids = predecessor_ids
        self._successor_ids = successor_ids

    layer = property(lambda self: MapLayer.LANE_GROUP)
    lane_ids = property(lambda self: self._lane_ids)
    intersection_id = property(lambda self: self._intersection_id)
    predecessor_ids = property(lambda self: self._predecessor_ids)
    successor_ids = property(lambda self: self._successor_ids)
    left_boundary = property(lambda self: self._left_boundary)
    right_boundary = property(lambda self: self._right_boundary)

    @property
    def successors(self) -> List[AbstractLaneGroup]:
        """Inherited, see superclass."""
        raise NotImplementedError

    @property
    def predecessors(self) -> List[AbstractLaneGroup]:
        """Inherited, see superclass."""
        raise NotImplementedError

    @property
    def lanes(self) -> List[AbstractLane]:
        """Inherited, see superclass."""
        raise NotImplementedError

    @property
    def intersection(self) -> Optional[AbstractIntersection]:
        """Inherited, see superclass."""
        raise NotImplementedError


class CacheIntersection(CacheSurfaceObject, AbstractIntersection):
    def __init__(
        self,
        object_id: MapObjectIDType,
        lane_group_ids: List[MapObjectIDType],
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):

        super().__init__(object_id, outline, geometry)
        self._lane_group_ids = lane_group_ids

    layer = property(lambda self: MapLayer.INTERSECTION)
    lane_group_ids = property(lambda self: self._lane_group_ids)

    @property
    def lane_groups(self) -> List[CacheLaneGroup]:
        """Inherited, see superclass."""
        raise NotImplementedError


class CacheCrosswalk(CacheSurfaceObject, AbstractCrosswalk):
    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)


class CacheCarpark(CacheSurfaceObject, AbstractCarpark):
    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)


class CacheWalkway(CacheSurfaceObject, AbstractWalkway):
    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)


class CacheGenericDrivable(CacheSurfaceObject, AbstractGenericDrivable):
    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)


class CacheRoadEdge(CacheLineObject, AbstractRoadEdge):
    def __init__(
        self,
        object_id: MapObjectIDType,
        road_edge_type: RoadEdgeType,
        polyline: Union[Polyline2D, Polyline3D],
    ):
        super().__init__(object_id, polyline)
        self._road_edge_type = road_edge_type

    road_edge_type = property(lambda self: self._road_edge_type)


class CacheRoadLine(CacheLineObject, AbstractRoadLine):
    def __init__(
        self,
        object_id: MapObjectIDType,
        road_line_type: RoadLineType,
        polyline: Union[Polyline2D, Polyline3D],
    ):
        super().__init__(object_id, polyline)
        self._road_line_type = road_line_type

    road_line_type = property(lambda self: self._road_line_type)
