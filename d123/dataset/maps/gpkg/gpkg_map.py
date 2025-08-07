from __future__ import annotations

import os
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import shapely
import shapely.geometry as geom

from d123.common.geometry.base import Point2D
from d123.dataset.maps.abstract_map import AbstractMap
from d123.dataset.maps.abstract_map_objects import AbstractMapObject
from d123.dataset.maps.gpkg.gpkg_map_objects import (
    GPKGCarpark,
    GPKGCrosswalk,
    GPKGGenericDrivable,
    GPKGIntersection,
    GPKGLane,
    GPKGLaneGroup,
    GPKGWalkway,
)
from d123.dataset.maps.gpkg.utils import load_gdf_with_geometry_columns
from d123.dataset.maps.map_datatypes import MapSurfaceType

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
            # MapSurfaceType.ROAD_EDGE: self._get_road_edge,
            # MapSurfaceType.ROAD_LINE: self._get_road_line,
        }

        # loaded during `.initialize()`
        self._gpd_dataframes: Dict[MapSurfaceType, gpd.GeoDataFrame] = {}

    @property
    def map_name(self) -> str:
        """Inherited, see superclass."""
        return self._file_path.with_suffix("").name

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if len(self._gpd_dataframes) == 0:
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
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapSurfaceType],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapSurfaceType, Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]]:
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapSurfaceType, Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]] = (
            defaultdict(list)
        )
        for layer in layers:
            object_map[layer] = self._query_layer(geometry, layer, predicate, sort, distance)
        return object_map

    def query_object_ids(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapSurfaceType],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapSurfaceType, Union[List[str], Dict[int, List[str]]]]:
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapSurfaceType, Union[List[str], Dict[int, List[str]]]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._query_layer_objects_ids(geometry, layer, predicate, sort, distance)
        return object_map

    def query_nearest(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapSurfaceType],
        return_all: bool = True,
        max_distance: Optional[float] = None,
        return_distance: bool = False,
        exclusive: bool = False,
    ) -> Dict[MapSurfaceType, Union[List[str], Dict[int, List[str]], Dict[int, List[Tuple[str, float]]]]]:
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapSurfaceType, Union[List[str], Dict[int, List[str]]]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._query_layer_nearest(
                geometry, layer, return_all, max_distance, return_distance, exclusive
            )
        return object_map

    def _query_layer(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layer: MapSurfaceType,
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]:
        queried_indices = self._gpd_dataframes[layer].sindex.query(
            geometry, predicate=predicate, sort=sort, distance=distance
        )

        if queried_indices.ndim == 2:
            query_dict: Dict[int, List[AbstractMapObject]] = defaultdict(list)
            for geometry_idx, map_object_idx in zip(queried_indices[0], queried_indices[1]):
                map_object_id = self._gpd_dataframes[layer]["id"].iloc[map_object_idx]
                query_dict[int(geometry_idx)].append(self.get_map_object(map_object_id, layer))
            return query_dict
        else:
            map_object_ids = self._gpd_dataframes[layer]["id"].iloc[queried_indices]
            return [self.get_map_object(map_object_id, layer) for map_object_id in map_object_ids]

    def _query_layer_objects_ids(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layer: MapSurfaceType,
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Union[List[str], Dict[int, List[str]]]:
        # Use numpy for fast indexing and avoid .iloc in a loop

        queried_indices = self._gpd_dataframes[layer].sindex.query(
            geometry, predicate=predicate, sort=sort, distance=distance
        )
        ids = self._gpd_dataframes[layer]["id"].values  # numpy array for fast access

        if queried_indices.ndim == 2:
            query_dict: Dict[int, List[str]] = defaultdict(list)
            for geometry_idx, map_object_idx in zip(queried_indices[0], queried_indices[1]):
                query_dict[int(geometry_idx)].append(ids[map_object_idx])
            return query_dict
        else:
            return list(ids[queried_indices])

    def _query_layer_nearest(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layer: MapSurfaceType,
        return_all: bool = True,
        max_distance: Optional[float] = None,
        return_distance: bool = False,
        exclusive: bool = False,
    ) -> Union[List[str], Dict[int, List[str]], Dict[int, List[Tuple[str, float]]]]:
        # Use numpy for fast indexing and avoid .iloc in a loop

        queried_indices = self._gpd_dataframes[layer].sindex.nearest(
            geometry,
            return_all=return_all,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
        )
        ids = self._gpd_dataframes[layer]["id"].values  # numpy array for fast access

        if return_distance:
            queried_indices, distances = queried_indices
            query_dict: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
            for geometry_idx, map_object_idx, distance in zip(queried_indices[0], queried_indices[1], distances):
                query_dict[int(geometry_idx)].append((ids[map_object_idx], float(distance)))
            return query_dict

        elif queried_indices.ndim == 2:
            query_dict: Dict[int, List[str]] = defaultdict(list)
            for geometry_idx, map_object_idx in zip(queried_indices[0], queried_indices[1]):
                query_dict[int(geometry_idx)].append(ids[map_object_idx])
            return query_dict
        else:
            return list(ids[queried_indices])

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

    # def _query_layer(
    #     self,
    #     geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
    #     layer: MapSurfaceType,
    #     predicate: Optional[str] = None,
    #     sort: bool = False,
    #     distance: Optional[float] = None,
    # ) -> Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]:
    #     queried_indices = self._gpd_dataframes[layer].sindex.query(
    #         geometry, predicate=predicate, sort=sort, distance=distance
    #     )
    #     ids = self._gpd_dataframes[layer]["id"].values  # numpy array for fast access
    #     if queried_indices.ndim == 2:
    #         query_dict: Dict[int, List[AbstractMapObject]] = defaultdict(list)
    #         for geometry_idx, map_object_idx in zip(queried_indices[0], queried_indices[1]):
    #             map_object_id = ids[map_object_idx]
    #             query_dict[int(geometry_idx)].append(self.get_map_object(map_object_id, layer))
    #         return query_dict
    #     else:
    #         map_object_ids = ids[queried_indices]
    #         return [self.get_map_object(map_object_id, layer) for map_object_id in map_object_ids]

    # def _query_layer_objects_ids(
    #     self,
    #     geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
    #     layer: MapSurfaceType,
    #     predicate: Optional[str] = None,
    #     sort: bool = False,
    #     distance: Optional[float] = None,
    # ) -> Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]:
    #     queried_indices = self._gpd_dataframes[layer].sindex.query(
    #         geometry, predicate=predicate, sort=sort, distance=distance
    #     )
    #     if queried_indices.ndim == 2:
    #         query_dict: Dict[int, List[AbstractMapObject]] = defaultdict(list)
    #         for geometry_idx, map_object_idx in zip(queried_indices[0], queried_indices[1]):
    #             map_object_id = self._gpd_dataframes[layer]["id"].iloc[map_object_idx]
    #             query_dict[int(geometry_idx)].append(map_object_id)
    #         return query_dict
    #     else:
    #         map_object_ids = self._gpd_dataframes[layer]["id"].iloc[queried_indices]
    #         return list(map_object_ids)


@lru_cache(maxsize=32)
def get_map_api_from_names(dataset: str, location: str) -> GPKGMap:
    D123_MAPS_ROOT = Path(os.environ.get("D123_MAPS_ROOT"))
    gpkg_path = D123_MAPS_ROOT / f"{dataset}_{location}.gpkg"
    assert gpkg_path.is_file(), f"{dataset}_{location}.gpkg not found in {str(D123_MAPS_ROOT)}."
    map_api = GPKGMap(gpkg_path)
    map_api.initialize()
    return map_api


def get_local_map_api(split_name: str, log_name: str) -> GPKGMap:
    print(split_name, log_name)
    D123_MAPS_ROOT = Path(os.environ.get("D123_MAPS_ROOT"))
    gpkg_path = D123_MAPS_ROOT / split_name / f"{log_name}.gpkg"
    assert gpkg_path.is_file(), f"{log_name}.gpkg not found in {str(D123_MAPS_ROOT)}."
    map_api = GPKGMap(gpkg_path)
    map_api.initialize()
    return map_api
