from __future__ import annotations

import ast
from functools import cached_property
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as geom
import trimesh

from d123.datatypes.maps.abstract_map_objects import (
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
from d123.datatypes.maps.gpkg.gpkg_utils import get_row_with_value, get_trimesh_from_boundaries
from d123.datatypes.maps.map_datatypes import RoadEdgeType, RoadLineType
from d123.geometry import Point3DIndex, Polyline3D


class GPKGSurfaceObject(AbstractSurfaceMapObject):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_id: MapObjectIDType, surface_df: gpd.GeoDataFrame) -> None:
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
        return get_row_with_value(self._object_df, "id", self.object_id)

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

        trimesh_mesh: Optional[trimesh.Trimesh] = None
        if "right_boundary" in self._object_df.columns and "left_boundary" in self._object_df.columns:
            left_boundary = Polyline3D.from_linestring(self._object_row.left_boundary)
            right_boundary = Polyline3D.from_linestring(self._object_row.right_boundary)
            trimesh_mesh = get_trimesh_from_boundaries(left_boundary, right_boundary)
        else:
            # Fallback to geometry if no boundaries are available
            outline_3d_array = self.outline_3d.array
            vertices_2d, faces = trimesh.creation.triangulate_polygon(
                geom.Polygon(outline_3d_array[:, Point3DIndex.XY])
            )
            if len(vertices_2d) == len(outline_3d_array):
                # Regular case, where vertices match outline_3d_array
                vertices_3d = outline_3d_array
            elif len(vertices_2d) == len(outline_3d_array) + 1:
                # outline array was not closed, so we need to add the first vertex again
                vertices_3d = np.vstack((outline_3d_array, outline_3d_array[0]))
            else:
                raise ValueError("No vertices found for triangulation.")
            trimesh_mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
        return trimesh_mesh


class GPKGLineObject(AbstractLineMapObject):

    def __init__(self, object_id: MapObjectIDType, line_df: gpd.GeoDataFrame) -> None:
        """
        Constructor of the base line map object type.
        :param object_id: unique identifier of a line map object.
        """
        super().__init__(object_id)
        # TODO: add assertion if columns are available
        self._object_df = line_df

    @cached_property
    def _object_row(self) -> gpd.GeoSeries:
        return get_row_with_value(self._object_df, "id", self.object_id)

    @property
    def polyline_3d(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.geometry)


class GPKGLane(GPKGSurfaceObject, AbstractLane):
    def __init__(
        self,
        object_id: MapObjectIDType,
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
    def successor_ids(self) -> List[MapObjectIDType]:
        """Inherited, see superclass."""
        return ast.literal_eval(self._object_row.successor_ids)

    @property
    def successors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        return [GPKGLane(lane_id, self._object_df) for lane_id in self.successor_ids]

    @property
    def predecessor_ids(self) -> List[MapObjectIDType]:
        """Inherited, see superclass."""
        return ast.literal_eval(self._object_row.predecessor_ids)

    @property
    def predecessors(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        return [GPKGLane(lane_id, self._object_df) for lane_id in self.predecessor_ids]

    @property
    def left_boundary(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.left_boundary)

    @property
    def right_boundary(self) -> Polyline3D:
        """Inherited, see superclass."""
        return Polyline3D.from_linestring(self._object_row.right_boundary)

    @property
    def left_lane_id(self) -> Optional[MapObjectIDType]:
        """ "Inherited, see superclass."""
        return self._object_row.left_lane_id

    @property
    def left_lane(self) -> Optional[GPKGLane]:
        """Inherited, see superclass."""
        return (
            GPKGLane(self.left_lane_id, self._object_df, self._lane_group_df, self._intersection_df)
            if self.left_lane_id is not None and not pd.isna(self.left_lane_id)
            else None
        )

    @property
    def right_lane_id(self) -> Optional[MapObjectIDType]:
        """Inherited, see superclass."""
        return self._object_row.right_lane_id

    @property
    def right_lane(self) -> Optional[GPKGLane]:
        """Inherited, see superclass."""
        return (
            GPKGLane(self.right_lane_id, self._object_df, self._lane_group_df, self._intersection_df)
            if self.right_lane_id is not None and not pd.isna(self.right_lane_id)
            else None
        )

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
    def lane_group_id(self) -> MapObjectIDType:
        """Inherited, see superclass."""
        return self._object_row.lane_group_id

    @property
    def lane_group(self) -> GPKGLaneGroup:
        """Inherited, see superclass."""
        return GPKGLaneGroup(
            self.lane_group_id,
            self._lane_group_df,
            self._object_df,
            self._intersection_df,
        )


class GPKGLaneGroup(GPKGSurfaceObject, AbstractLaneGroup):
    def __init__(
        self,
        object_id: MapObjectIDType,
        object_df: gpd.GeoDataFrame,
        lane_df: gpd.GeoDataFrame,
        intersection_df: gpd.GeoDataFrame,
    ):
        super().__init__(object_id, object_df)
        self._lane_df = lane_df
        self._intersection_df = intersection_df

    @property
    def successor_ids(self) -> List[MapObjectIDType]:
        """Inherited, see superclass."""
        return ast.literal_eval(self._object_row.successor_ids)

    @property
    def successors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        return [
            GPKGLaneGroup(lane_group_id, self._object_df, self._lane_df, self._intersection_df)
            for lane_group_id in self.successor_ids
        ]

    @property
    def predecessor_ids(self) -> List[MapObjectIDType]:
        """Inherited, see superclass."""
        return ast.literal_eval(self._object_row.predecessor_ids)

    @property
    def predecessors(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        return [
            GPKGLaneGroup(lane_group_id, self._object_df, self._lane_df, self._intersection_df)
            for lane_group_id in self.predecessor_ids
        ]

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
    def lane_ids(self) -> List[MapObjectIDType]:
        """Inherited, see superclass."""
        return ast.literal_eval(self._object_row.lane_ids)

    @property
    def lanes(self) -> List[GPKGLane]:
        """Inherited, see superclass."""
        return [
            GPKGLane(
                lane_id,
                self._lane_df,
                self._object_df,
                self._intersection_df,
            )
            for lane_id in self.lane_ids
        ]

    @property
    def intersection_id(self) -> Optional[MapObjectIDType]:
        """Inherited, see superclass."""
        return self._object_row.intersection_id

    @property
    def intersection(self) -> Optional[GPKGIntersection]:
        """Inherited, see superclass."""
        return (
            GPKGIntersection(
                self.intersection_id,
                self._intersection_df,
                self._lane_df,
                self._object_df,
            )
            if self.intersection_id is not None and not pd.isna(self.intersection_id)
            else None
        )


class GPKGIntersection(GPKGSurfaceObject, AbstractIntersection):
    def __init__(
        self,
        object_id: MapObjectIDType,
        object_df: gpd.GeoDataFrame,
        lane_df: gpd.GeoDataFrame,
        lane_group_df: gpd.GeoDataFrame,
    ):
        super().__init__(object_id, object_df)
        self._lane_df = lane_df
        self._lane_group_df = lane_group_df

    @property
    def lane_group_ids(self) -> List[MapObjectIDType]:
        """Inherited, see superclass."""
        return ast.literal_eval(self._object_row.lane_group_ids)

    @property
    def lane_groups(self) -> List[GPKGLaneGroup]:
        """Inherited, see superclass."""
        return [
            GPKGLaneGroup(
                lane_group_id,
                self._lane_group_df,
                self._lane_df,
                self._object_df,
            )
            for lane_group_id in self.lane_group_ids
        ]


class GPKGCrosswalk(GPKGSurfaceObject, AbstractCrosswalk):
    def __init__(self, object_id: MapObjectIDType, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGCarpark(GPKGSurfaceObject, AbstractCarpark):
    def __init__(self, object_id: MapObjectIDType, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGWalkway(GPKGSurfaceObject, AbstractWalkway):
    def __init__(self, object_id: MapObjectIDType, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGGenericDrivable(GPKGSurfaceObject, AbstractGenericDrivable):
    def __init__(self, object_id: MapObjectIDType, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)


class GPKGRoadEdge(GPKGLineObject, AbstractRoadEdge):
    def __init__(self, object_id: MapObjectIDType, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)

    @cached_property
    def _object_row(self) -> gpd.GeoSeries:
        return get_row_with_value(self._object_df, "id", self.object_id)

    @property
    def road_edge_type(self) -> RoadEdgeType:
        """Inherited, see superclass."""
        return RoadEdgeType(int(self._object_row.road_edge_type))


class GPKGRoadLine(GPKGLineObject, AbstractRoadLine):
    def __init__(self, object_id: MapObjectIDType, object_df: gpd.GeoDataFrame):
        super().__init__(object_id, object_df)

    @cached_property
    def _object_row(self) -> gpd.GeoSeries:
        return get_row_with_value(self._object_df, "id", self.object_id)

    @property
    def road_line_type(self) -> RoadLineType:
        """Inherited, see superclass."""
        return RoadLineType(int(self._object_row.road_line_type))
