from __future__ import annotations

import os
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import geopandas as gpd
import shapely
import shapely.geometry as geom

from asim.common.geometry.base import Point2D
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.abstract_map_objects import AbstractMapObject
from asim.dataset.maps.gpkg.gpkg_map_objects import (
    GPKGCarpark,
    GPKGCrosswalk,
    GPKGGenericDrivable,
    GPKGIntersection,
    GPKGLane,
    GPKGLaneGroup,
    GPKGWalkway,
)
from asim.dataset.maps.gpkg.utils import load_gdf_with_geometry_columns
from asim.dataset.maps.map_datatypes import MapSurfaceType

USE_ARROW: bool = True


class GPKGMap(AbstractMap):
    def __init__(self, file_path: Path) -> None:

        self._file_path = file_path
        self._map_object_getter: Dict[MapSurfaceType, Callable[[str], Optional[AbstractMapObject]]] = {
            MapSurfaceType.LANE: self._get_lane,
            MapSurfaceType.LANE_GROUP: self._get_lane_group,
            MapSurfaceType.INTERSECTION: self._get_intersection,
            MapSurfaceType.CROSSWALK: self._get_crosswalk,
            MapSurfaceType.WALKWAY: self._get_walkway,
            MapSurfaceType.CARPARK: self._get_carpark,
            MapSurfaceType.GENERIC_DRIVABLE: self._get_generic_drivable,
        }

        # loaded during `.initialize()`
        self._gpd_dataframes: Dict[MapSurfaceType, gpd.GeoDataFrame] = {}

    @property
    def map_name(self) -> str:
        """Inherited, see superclass."""
        return self._file_path.with_suffix("").name

    def initialize(self) -> None:
        """Inherited, see superclass."""

        available_layers = list(gpd.list_layers(self._file_path).name)
        for map_layer in list(MapSurfaceType):
            map_layer_name = map_layer.serialize()
            if map_layer_name in available_layers:
                self._gpd_dataframes[map_layer] = gpd.read_file(
                    self._file_path, layer=map_layer_name, use_arrow=USE_ARROW
                )
                load_gdf_with_geometry_columns(
                    self._gpd_dataframes[map_layer],
                    geometry_column_names=["baseline_path", "right_boundary", "left_boundary", "outline"],
                )
                # TODO: remove the temporary fix and enforce consistent id types in the GPKG files
                if "id" in self._gpd_dataframes[map_layer].columns:
                    self._gpd_dataframes[map_layer]["id"] = self._gpd_dataframes[map_layer]["id"].astype(str)
            else:
                warnings.warn(f"GPKGMap: {map_layer_name} not available in {str(self._file_path)}")

    def _assert_initialize(self) -> None:
        "Checks if `.initialize()` was called, before retrieving data."
        assert len(self._gpd_dataframes) > 0, "GPKGMap: Call `.initialize()` before retrieving data!"

    def _assert_layer_available(self, layer: MapSurfaceType) -> None:
        "Checks if layer is available."
        assert layer in self.get_available_map_objects(), f"GPKGMap: MapSurfaceType {layer.name} is unavailable."

    def get_available_map_objects(self) -> List[MapSurfaceType]:
        """Inherited, see superclass."""
        self._assert_initialize()
        return list(self._gpd_dataframes.keys())

    def get_map_object(self, object_id: str, layer: MapSurfaceType) -> Optional[AbstractMapObject]:
        """Inherited, see superclass."""

        self._assert_initialize()
        self._assert_layer_available(layer)
        try:
            return self._map_object_getter[layer](object_id)
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} object: {object_id} is unavailable")

    def get_all_map_objects(self, point_2d: Point2D, layer: MapSurfaceType) -> List[AbstractMapObject]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def is_in_layer(self, point: Point2D, layer: MapSurfaceType) -> bool:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_proximal_map_objects(
        self, point_2d: Point2D, radius: float, layers: List[MapSurfaceType]
    ) -> Dict[MapSurfaceType, List[AbstractMapObject]]:
        """Inherited, see superclass."""
        center_point = geom.Point(point_2d.x, point_2d.y)
        patch = center_point.buffer(radius)
        return self.query(geometry=patch, layers=layers, predicate="intersects")

    def query(
        self,
        geometry: shapely.Geometry,
        layers: List[MapSurfaceType],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapSurfaceType, List[AbstractMapObject]]:
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapSurfaceType, List[AbstractMapObject]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._query_layer(geometry, layer, predicate, sort, distance)
        return object_map

    def _query_layer(
        self,
        geometry: shapely.Geometry,
        layer: MapSurfaceType,
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> List[AbstractMapObject]:
        queried_indices = list(
            self._gpd_dataframes[layer].sindex.query(geometry, predicate=predicate, sort=sort, distance=distance)
        )
        map_object_ids = self._gpd_dataframes[layer].iloc[queried_indices]["id"]
        return [self.get_map_object(map_object_id, layer) for map_object_id in map_object_ids]

    def _get_lane(self, id: str) -> Optional[GPKGLane]:
        return (
            GPKGLane(
                id,
                self._gpd_dataframes[MapSurfaceType.LANE],
                self._gpd_dataframes[MapSurfaceType.LANE_GROUP],
                self._gpd_dataframes[MapSurfaceType.INTERSECTION],
            )
            if id in self._gpd_dataframes[MapSurfaceType.LANE]["id"].tolist()
            else None
        )

    def _get_lane_group(self, id: str) -> Optional[GPKGLaneGroup]:
        return (
            GPKGLaneGroup(
                id,
                self._gpd_dataframes[MapSurfaceType.LANE_GROUP],
                self._gpd_dataframes[MapSurfaceType.LANE],
                self._gpd_dataframes[MapSurfaceType.INTERSECTION],
            )
            if id in self._gpd_dataframes[MapSurfaceType.LANE_GROUP]["id"].tolist()
            else None
        )

    def _get_intersection(self, id: str) -> Optional[GPKGIntersection]:
        return (
            GPKGIntersection(
                id,
                self._gpd_dataframes[MapSurfaceType.INTERSECTION],
                self._gpd_dataframes[MapSurfaceType.LANE],
                self._gpd_dataframes[MapSurfaceType.LANE_GROUP],
            )
            if id in self._gpd_dataframes[MapSurfaceType.INTERSECTION]["id"].tolist()
            else None
        )

    def _get_crosswalk(self, id: str) -> Optional[GPKGCrosswalk]:
        return (
            GPKGCrosswalk(id, self._gpd_dataframes[MapSurfaceType.CROSSWALK])
            if id in self._gpd_dataframes[MapSurfaceType.CROSSWALK]["id"].tolist()
            else None
        )

    def _get_walkway(self, id: str) -> Optional[GPKGWalkway]:
        return (
            GPKGWalkway(id, self._gpd_dataframes[MapSurfaceType.WALKWAY])
            if id in self._gpd_dataframes[MapSurfaceType.WALKWAY]["id"].tolist()
            else None
        )

    def _get_carpark(self, id: str) -> Optional[GPKGCarpark]:
        return (
            GPKGCarpark(id, self._gpd_dataframes[MapSurfaceType.CARPARK])
            if id in self._gpd_dataframes[MapSurfaceType.CARPARK]["id"].tolist()
            else None
        )

    def _get_generic_drivable(self, id: str) -> Optional[GPKGGenericDrivable]:
        return (
            GPKGGenericDrivable(id, self._gpd_dataframes[MapSurfaceType.GENERIC_DRIVABLE])
            if id in self._gpd_dataframes[MapSurfaceType.GENERIC_DRIVABLE]["id"].tolist()
            else None
        )


@lru_cache(maxsize=32)
def get_map_api_from_names(dataset: str, location: str) -> GPKGMap:
    ASIM_MAPS_ROOT = Path(os.environ.get("ASIM_MAPS_ROOT"))
    gpkg_path = ASIM_MAPS_ROOT / f"{dataset}_{location}.gpkg"
    assert gpkg_path.is_file(), f"{dataset}_{location}.gpkg not found in {str(ASIM_MAPS_ROOT)}."
    map_api = GPKGMap(gpkg_path)
    map_api.initialize()
    return map_api
