from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import shapely.geometry as geom
from trimesh import Trimesh

from py123d.datatypes.map_objects.base_map_objects import BaseMapLineObject, BaseMapSurfaceObject, MapObjectIDType
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.map_objects.utils import get_trimesh_from_boundaries
from py123d.geometry import Polyline2D, Polyline3D

if TYPE_CHECKING:
    from py123d.api.map.map_api import MapAPI


class Lane(BaseMapSurfaceObject):

    __slots__ = (
        "_lane_group_id",
        "_left_boundary",
        "_right_boundary",
        "_centerline",
        "_left_lane_id",
        "_right_lane_id",
        "_predecessor_ids",
        "_successor_ids",
        "_speed_limit_mps",
        "_map_api",
    )

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
        map_api: Optional["MapAPI"] = None,
    ) -> None:

        if outline is None:
            outline_array = np.vstack(
                (
                    left_boundary.array,
                    right_boundary.array[::-1],
                    left_boundary.array[0],
                )
            )
            outline = Polyline3D.from_array(outline_array)

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
        self._map_api = map_api

    @property
    def layer(self) -> MapLayer:
        return MapLayer.LANE

    @property
    def lane_group_id(self) -> MapObjectIDType:
        return self._lane_group_id

    @property
    def lane_group(self) -> Optional[LaneGroup]:
        if self._map_api is not None:
            return self._map_api.get_map_object(self.lane_group_id, MapLayer.LANE_GROUP)
        return None

    @property
    def left_boundary(self) -> Polyline3D:
        return self._left_boundary

    @property
    def right_boundary(self) -> Polyline3D:
        return self._right_boundary

    @property
    def centerline(self) -> Polyline3D:
        return self._centerline

    @property
    def left_lane_id(self) -> Optional[MapObjectIDType]:
        return self._left_lane_id

    @property
    def left_lane(self) -> Optional[Lane]:
        if self._map_api is not None and self.left_lane_id is not None:
            return self._map_api.get_map_object(self.left_lane_id, self.layer)
        return None

    @property
    def right_lane_id(self) -> Optional[MapObjectIDType]:
        return self._right_lane_id

    @property
    def right_lane(self) -> Optional[Lane]:
        if self._map_api is not None and self.right_lane_id is not None:
            return self._map_api.get_map_object(self.right_lane_id, self.layer)
        return None

    @property
    def predecessor_ids(self) -> List[MapObjectIDType]:
        return self._predecessor_ids

    @property
    def predecessors(self) -> List[Lane]:
        predecessors: Optional[List[Lane]] = None
        if self._map_api is not None:
            predecessors = [self._map_api.get_map_object(lane_id, self.layer) for lane_id in self.predecessor_ids]
        return predecessors

    @property
    def successor_ids(self) -> List[MapObjectIDType]:
        return self._successor_ids

    @property
    def successors(self) -> List[Lane]:
        successors: Optional[List[Lane]] = None
        if self._map_api is not None:
            successors = [self._map_api.get_map_object(lane_id, self.layer) for lane_id in self.successor_ids]
        return successors

    @property
    def speed_limit_mps(self) -> Optional[float]:
        return self._speed_limit_mps

    @property
    def trimesh_mesh(self) -> Trimesh:
        return get_trimesh_from_boundaries(self.left_boundary, self.right_boundary)


class LaneGroup(BaseMapSurfaceObject):

    __slots__ = (
        "_lane_ids",
        "_left_boundary",
        "_right_boundary",
        "_intersection_id",
        "_predecessor_ids",
        "_successor_ids",
        "_map_api",
    )

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
        map_api: Optional["MapAPI"] = None,
    ):
        if outline is None:
            outline_array = np.vstack(
                (
                    left_boundary.array,
                    right_boundary.array[::-1],
                    left_boundary.array[0],
                )
            )
            outline = Polyline3D.from_array(outline_array)
        super().__init__(object_id, outline, geometry)

        self._lane_ids = lane_ids
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._intersection_id = intersection_id
        self._predecessor_ids = predecessor_ids
        self._successor_ids = successor_ids
        self._map_api = map_api

    @property
    def layer(self) -> MapLayer:
        return MapLayer.LANE_GROUP

    @property
    def lane_ids(self) -> List[MapObjectIDType]:
        return self._lane_ids

    @property
    def lanes(self) -> List[Lane]:
        lanes: Optional[List[Lane]] = None
        if self._map_api is not None:
            lanes = [self._map_api.get_map_object(lane_id, MapLayer.LANE) for lane_id in self.lane_ids]
        return lanes

    @property
    def left_boundary(self) -> Polyline3D:
        return self._left_boundary

    @property
    def right_boundary(self) -> Polyline3D:
        return self._right_boundary

    @property
    def intersection_id(self) -> Optional[MapObjectIDType]:
        return self._intersection_id

    @property
    def intersection(self) -> Optional[Intersection]:
        if self._map_api is not None and self.intersection_id is not None:
            return self._map_api.get_map_object(self.intersection_id, MapLayer.INTERSECTION)
        return None

    @property
    def predecessor_ids(self) -> List[MapObjectIDType]:
        return self._predecessor_ids

    @property
    def predecessors(self) -> List[LaneGroup]:
        predecessors: Optional[List[LaneGroup]] = None
        if self._map_api is not None:
            predecessors = [
                self._map_api.get_map_object(lane_group_id, self.layer) for lane_group_id in self.predecessor_ids
            ]
        return predecessors

    @property
    def successor_ids(self) -> List[MapObjectIDType]:
        return self._successor_ids

    @property
    def successors(self) -> List[LaneGroup]:
        successors: Optional[List[LaneGroup]] = None
        if self._map_api is not None:
            successors = [
                self._map_api.get_map_object(lane_group_id, self.layer) for lane_group_id in self.successor_ids
            ]
        return successors

    @property
    def trimesh_mesh(self) -> Trimesh:
        return get_trimesh_from_boundaries(self.left_boundary, self.right_boundary)


class Intersection(BaseMapSurfaceObject):

    __slots__ = (
        "_lane_group_ids",
        "_map_api",
    )

    def __init__(
        self,
        object_id: MapObjectIDType,
        lane_group_ids: List[MapObjectIDType],
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
        map_api: Optional["MapAPI"] = None,
    ):
        super().__init__(object_id, outline, geometry)
        self._lane_group_ids = lane_group_ids
        self._map_api = map_api

    @property
    def layer(self) -> MapLayer:
        return MapLayer.INTERSECTION

    @property
    def lane_group_ids(self) -> List[MapObjectIDType]:
        return self._lane_group_ids

    @property
    def lane_groups(self) -> List[LaneGroup]:
        lane_groups: Optional[List[LaneGroup]] = None
        if self._map_api is not None:
            lane_groups = [
                self._map_api.get_map_object(lane_group_id, MapLayer.LANE_GROUP)
                for lane_group_id in self.lane_group_ids
            ]
        return lane_groups


class Crosswalk(BaseMapSurfaceObject):

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)

    @property
    def layer(self) -> MapLayer:
        return MapLayer.CROSSWALK


class Carpark(BaseMapSurfaceObject):

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)

    @property
    def layer(self) -> MapLayer:
        return MapLayer.CARPARK


class Walkway(BaseMapSurfaceObject):

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)

    @property
    def layer(self) -> MapLayer:
        return MapLayer.WALKWAY


class GenericDrivable(BaseMapSurfaceObject):

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)

    @property
    def layer(self) -> MapLayer:
        return MapLayer.GENERIC_DRIVABLE


class StopZone(BaseMapSurfaceObject):

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ):
        super().__init__(object_id, outline, geometry)

    @property
    def layer(self) -> MapLayer:
        return MapLayer.STOP_ZONE


class RoadEdge(BaseMapLineObject):

    __slots__ = ("_road_edge_type",)

    def __init__(
        self,
        object_id: MapObjectIDType,
        road_edge_type: int,
        polyline: Union[Polyline2D, Polyline3D],
    ):
        super().__init__(object_id, polyline)
        self._road_edge_type = road_edge_type

    @property
    def layer(self) -> MapLayer:
        return MapLayer.ROAD_EDGE

    @property
    def road_edge_type(self) -> int:
        return self._road_edge_type


class RoadLine(BaseMapLineObject):

    __slots__ = ("_road_line_type",)

    def __init__(
        self,
        object_id: MapObjectIDType,
        road_line_type: int,
        polyline: Union[Polyline2D, Polyline3D],
    ):
        super().__init__(object_id, polyline)
        self._road_line_type = road_line_type

    @property
    def layer(self) -> MapLayer:
        return MapLayer.ROAD_LINE

    @property
    def road_line_type(self) -> int:
        return self._road_line_type
