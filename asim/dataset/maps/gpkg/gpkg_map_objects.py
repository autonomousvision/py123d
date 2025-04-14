from __future__ import annotations

from functools import cached_property
from typing import List, Optional
import geopandas as gpd


import shapely.geometry as geom

from asim.dataset.maps.abstract_map_objects import (
    AbstractLane,
    AbstractLaneGroup,
    AbstractIntersection,
    AbstractCrosswalk,
    AbstractCarpark,
    AbstractGenericDrivable,
)
from asim.dataset.maps.gpkg.utils import get_row_with_value


# "id", "predecessor_ids", "successor_ids", "left_boundary", "right_boundary", "baseline_path", "geometry"
class GPKGLane(AbstractLane):
    def __init__(self, object_id: str, lanes_df: gpd.GeoDataFrame):
        """
        Constructor of the gpkg lane type.
        :param object_id: unique identifier of the lane.
        :param lanes_df: geopandas lane dataframe.
        """
        super().__init__(object_id)

        self._lanes_df = lanes_df

    @property
    def successors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        pass

    @property
    def predecessors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        pass

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """Inherited, see superclass."""
        return self._lane_row.geometry

    @cached_property
    def _lane_row(self) -> gpd.GeoSeries:
        return get_row_with_value(self._lanes_df, "id", self.id)


# "lane_group_id", "predecessor_lane_group_id", "successor_lane_group_id", "left_boundary", "right_boundary", "geometry"
class GPKGLaneGroup(AbstractLaneGroup):
    def __init__(self, object_id: str, lane_groups_df: gpd.GeoDataFrame):
        """
        Constructor of the gpkg lane group type.
        :param object_id: unique identifier of the lane group.
        :param lanes_df: geopandas lane group dataframe.
        """
        super().__init__(object_id)

        self._lane_groups_df = lane_groups_df
        self._lane_group_row: Optional[gpd.GeoSeries] = None

    @property
    def successors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        pass

    @property
    def predecessors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        pass

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """Inherited, see superclass."""
        return self._lane_group_row().geometry

    @cached_property
    def _lane_group_row(self) -> gpd.GeoSeries:
        return get_row_with_value(self._lane_groups_df, "id", self.id)


class GPKGIntersection(AbstractIntersection):
    pass


class GPKGCrosswalk(AbstractCrosswalk):
    pass


class GPKGCarpark(AbstractCarpark):
    pass


class GPKGGenericDrivable(AbstractGenericDrivable):
    pass
