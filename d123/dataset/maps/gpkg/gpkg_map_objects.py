from __future__ import annotations

import ast
from functools import cached_property
from typing import List, Optional

import geopandas as gpd
import numpy as np
import shapely.geometry as geom
import trimesh

from d123.common.geometry.base import Point3DIndex
from d123.common.geometry.line.polylines import Polyline3D
from d123.common.visualization.viser.utils import get_trimesh_from_boundaries
from d123.dataset.maps.abstract_map_objects import (
    AbstractCarpark,
    AbstractCrosswalk,
    AbstractGenericDrivable,
    AbstractIntersection,
    AbstractLane,
    AbstractLaneGroup,
    AbstractSurfaceMapObject,
    AbstractWalkway,
)
from d123.dataset.maps.gpkg.utils import get_row_with_value


class GPKGSurfaceObject(AbstractSurfaceMapObject):
    """
    Base interface representation of all map objects.
    """

    # TODO: Extend for 3D outline
    def __init__(self, object_id: str, surface_df: gpd.GeoDataFrame) -> None:
        """
        Constructor of the base surface map object type.
        :param object_id: unique identifier of a surface map object.
        """
        super().__init__(object_id)
        # TODO: add assertion if columns are available
        self._object_df = surface_df

    @property
    def shapely_polygon(self) -> geom.Polygon:
        """Inherited, see superclass."""
        return self._object_row.geometry

    @cached_property
    def _object_row(self) -> gpd.GeoSeries:
        return get_row_with_value(self._object_df, "id", self.id)

    @cached_property
    def outline_3d(self) -> Polyline3D:
        """Inherited, see superclass."""
        outline_3d: Optional[Polyline3D] = None
        if "outline" in self._object_df.columns:
            outline_3d = Polyline3D.from_linestring(self._object_row.outline)
        else:
            outline_3d = Polyline3D.from_linestring(geom.LineString(self.shapely_polygon.exterior.coords))
        return outline_3d

    @property
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """Inherited, see superclass."""
        outline_3d_array = self.outline_3d.array
        _, faces = trimesh.creation.triangulate_polygon(geom.Polygon(outline_3d_array[:, Point3DIndex.XY]))
        # NOTE: Optional add color information to the mesh
        return trimesh.Trimesh(vertices=outline_3d_array, faces=faces)


class GPKGLane(GPKGSurfaceObject, AbstractLane):
    def __init__(
        self,
        object_id: str,
        object_df: gpd.GeoDataFrame,
        lane_group_df: gpd.GeoDataFrame,
        intersection_df: gpd.GeoDataFrame,
    ) -> None:
        super().__init__(object_id, object_df)
        self._lane_group_df = lane_group_df
        self._intersection_df = intersection_df

    @property
    def speed_limit_mps(self) -> Optional[float]:
        """Inherited, see superclass."""
        return self._object_row.speed_limit_mps

    @property
    def successors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        successor_ids = ast.literal_eval(self._object_row.successor_ids)
        return [GPKGLane(lane_id, self._object_df) for lane_id in successor_ids]

    @property
    def predecessors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        predecessor_ids = ast.literal_eval(self._object_row.predecessor_ids)
        return [GPKGLane(lane_id, self._object_df) for lane_id in predecessor_ids]

    @property
    def left_boundary(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.left_boundary)

    @property
    def right_boundary(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.right_boundary)

    @property
    def centerline(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.baseline_path)

    @property
    def outline_3d(self) -> Polyline3D:
        """Inherited, see superclass."""
        outline_array = np.vstack((self.left_boundary.array, self.right_boundary.array[::-1]))
        outline_array = np.vstack((outline_array, outline_array[0]))
        return Polyline3D.from_linestring(geom.LineString(outline_array))

    @property
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """Inherited, see superclass."""
        return get_trimesh_from_boundaries(self.left_boundary, self.right_boundary)

    @property
    def lane_group(self) -> GPKGLaneGroup:
        """Inherited, see superclass."""
        lane_group_id = self._object_row.lane_group_id
        return GPKGLaneGroup(
            lane_group_id,
            self._lane_group_df,
            self._object_df,
            self._intersection_df,
        )


class GPKGLaneGroup(GPKGSurfaceObject, AbstractLaneGroup):
    def __init__(
        self,
        object_id: str,
        object_df: gpd.GeoDataFrame,
        lane_df: gpd.GeoDataFrame,
        intersection_df: gpd.GeoDataFrame,
    ):
        super().__init__(object_id, object_df)
        self._lane_df = lane_df
        self._intersection_df = intersection_df

    @property
    def successors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        successor_ids = ast.literal_eval(self._object_row.successor_ids)
        return [GPKGLaneGroup(lane_group_id, self._object_df) for lane_group_id in successor_ids]

    @property
    def predecessors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        predecessor_ids = ast.literal_eval(self._object_row.predecessor_ids)
        return [GPKGLaneGroup(lane_group_id, self._object_df) for lane_group_id in predecessor_ids]

    @property
    def left_boundary(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.left_boundary)

    @property
    def right_boundary(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.right_boundary)

    @property
    def outline_3d(self) -> Polyline3D:
        """Inherited, see superclass."""
        outline_array = np.vstack((self.left_boundary.array, self.right_boundary.array[::-1]))
        return Polyline3D.from_linestring(geom.LineString(outline_array))

    @property
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """Inherited, see superclass."""
        return get_trimesh_from_boundaries(self.left_boundary, self.right_boundary)

    @property
    def lanes(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        lane_ids = ast.literal_eval(self._object_row.lane_ids)
        return [
            GPKGLane(
                lane_id,
                self._lane_df,
                self._object_df,
                self._intersection_df,
            )
            for lane_id in lane_ids
        ]

    @property
    def intersection(self) -> Optional[GPKGIntersection]:
        """Inherited, see superclass."""
        intersection_id = self._object_row.intersection_id
        return (
            GPKGIntersection(
                intersection_id,
                self._intersection_df,
                self._lane_df,
                self._object_df,
            )
            if intersection_id is not None
            else None
        )


class GPKGIntersection(GPKGSurfaceObject, AbstractIntersection):
    def __init__(
        self,
        object_id: str,
        object_df: gpd.GeoDataFrame,
        lane_df: gpd.GeoDataFrame,
        lane_group_df: gpd.GeoDataFrame,
    ):
        super().__init__(object_id, object_df)
        self._lane_df = lane_df
        self._lane_group_df = lane_group_df

    @property
    def lane_groups(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        lane_group_ids = ast.literal_eval(self._object_row.lane_group_ids)
        return [
            GPKGLaneGroup(
                lane_group_id,
                self._lane_group_df,
                self._lane_df,
                self._object_df,
            )
            for lane_group_id in lane_group_ids
        ]


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
