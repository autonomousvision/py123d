from __future__ import annotations

from functools import cached_property
from typing import List, Optional
import geopandas as gpd
import ast


import shapely.geometry as geom

from asim.dataset.maps.abstract_map_objects import (
    AbstractSurfaceMapObject,
    AbstractLane,
    AbstractLaneGroup,
    AbstractIntersection,
    AbstractCrosswalk,
    AbstractWalkway,
    AbstractCarpark,
    AbstractGenericDrivable,
)
from asim.dataset.maps.gpkg.utils import get_row_with_value


class GPKGSurfaceObject(AbstractSurfaceMapObject):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_id: str, surface_df: gpd.GeoDataFrame) -> None:
        """
        Constructor of the base surface map object type.
        :param object_id: unique identifier of a surface map object.
        """
        super().__init__(object_id)
        # TODO: add assertion if columns are available
        self._surface_df = surface_df

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """Inherited, see superclass."""
        return self._object_row.geometry

    @cached_property
    def _object_row(self) -> gpd.GeoSeries:
        return get_row_with_value(self._surface_df, "id", self.id)


# "id", "predecessor_ids", "successor_ids", "left_boundary", "right_boundary", "baseline_path", "geometry"
class GPKGLane(GPKGSurfaceObject, AbstractLane):

    def __init__(self, object_id: str, object_df: gpd.GeoDataFrame) -> None:
        super().__init__(object_id, object_df)

    @property
    def successors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        successor_ids = ast.literal_eval(self._object_row.successor_ids)
        return [GPKGLane(lane_id, self._surface_df) for lane_id in successor_ids]

    @property
    def predecessors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        predecessor_ids = ast.literal_eval(self._object_row.predecessor_ids)
        return [GPKGLane(lane_id, self._surface_df) for lane_id in predecessor_ids]


# "lane_group_id", "predecessor_lane_group_id", "successor_lane_group_id", "left_boundary", "right_boundary", "geometry"
class GPKGLaneGroup(GPKGSurfaceObject, AbstractLaneGroup):

    def __init__(self, object_id: str, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)

    @property
    def successors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        successor_ids = ast.literal_eval(self._object_row.successor_ids)
        return [GPKGLane(lane_group_id, self._surface_df) for lane_group_id in successor_ids]

    @property
    def predecessors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        predecessor_ids = ast.literal_eval(self._object_row.predecessor_ids)
        return [GPKGLane(lane_group_id, self._surface_df) for lane_group_id in predecessor_ids]


class GPKGIntersection(GPKGSurfaceObject, AbstractIntersection):

    def __init__(self, object_id: str, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGCrosswalk(GPKGSurfaceObject, AbstractCrosswalk):

    def __init__(self, object_id: str, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGCarpark(GPKGSurfaceObject, AbstractCarpark):

    def __init__(self, object_id: str, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGWalkway(GPKGSurfaceObject, AbstractWalkway):

    def __init__(self, object_id: str, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGGenericDrivable(GPKGSurfaceObject, AbstractGenericDrivable):

    def __init__(self, object_id: str, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)
