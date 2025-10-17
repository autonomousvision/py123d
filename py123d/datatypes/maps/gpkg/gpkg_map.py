from __future__ import annotations

import os
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Final, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import shapely
import shapely.geometry as geom

from py123d.datatypes.maps.abstract_map import AbstractMap
from py123d.datatypes.maps.abstract_map_objects import AbstractMapObject
from py123d.datatypes.maps.gpkg.gpkg_map_objects import (
    GPKGCarpark,
    GPKGCrosswalk,
    GPKGGenericDrivable,
    GPKGIntersection,
    GPKGLane,
    GPKGLaneGroup,
    GPKGRoadEdge,
    GPKGRoadLine,
    GPKGWalkway,
)
from py123d.datatypes.maps.gpkg.gpkg_utils import load_gdf_with_geometry_columns
from py123d.datatypes.maps.map_datatypes import MapLayer
from py123d.datatypes.maps.map_metadata import MapMetadata
from py123d.geometry import Point2D

USE_ARROW: bool = True
MAX_LRU_CACHED_TABLES: Final[int] = 128  # TODO: add to some configs


class GPKGMap(AbstractMap):
    def __init__(self, file_path: Path) -> None:

        self._file_path = file_path
        self._map_object_getter: Dict[MapLayer, Callable[[str], Optional[AbstractMapObject]]] = {
            MapLayer.LANE: self._get_lane,
            MapLayer.LANE_GROUP: self._get_lane_group,
            MapLayer.INTERSECTION: self._get_intersection,
            MapLayer.CROSSWALK: self._get_crosswalk,
            MapLayer.WALKWAY: self._get_walkway,
            MapLayer.CARPARK: self._get_carpark,
            MapLayer.GENERIC_DRIVABLE: self._get_generic_drivable,
            MapLayer.ROAD_EDGE: self._get_road_edge,
            MapLayer.ROAD_LINE: self._get_road_line,
        }

        # loaded during `.initialize()`
        self._gpd_dataframes: Dict[MapLayer, gpd.GeoDataFrame] = {}
        self._map_metadata: Optional[MapMetadata] = None

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if len(self._gpd_dataframes) == 0:
            available_layers = list(gpd.list_layers(self._file_path).name)
            for map_layer in list(MapLayer):
                map_layer_name = map_layer.serialize()
                if map_layer_name in available_layers:
                    self._gpd_dataframes[map_layer] = gpd.read_file(
                        self._file_path, layer=map_layer_name, use_arrow=USE_ARROW
                    )
                    load_gdf_with_geometry_columns(
                        self._gpd_dataframes[map_layer],
                        geometry_column_names=["centerline", "right_boundary", "left_boundary", "outline"],
                    )
                    # TODO: remove the temporary fix and enforce consistent id types in the GPKG files
                    if "id" in self._gpd_dataframes[map_layer].columns:
                        self._gpd_dataframes[map_layer]["id"] = self._gpd_dataframes[map_layer]["id"].astype(str)
                else:
                    warnings.warn(f"GPKGMap: {map_layer_name} not available in {str(self._file_path)}")
                    self._gpd_dataframes[map_layer] = None

            assert "map_metadata" in list(gpd.list_layers(self._file_path).name)
            metadata_gdf = gpd.read_file(self._file_path, layer="map_metadata", use_arrow=USE_ARROW)
            self._map_metadata = MapMetadata.from_dict(metadata_gdf.iloc[0].to_dict())

    def _assert_initialize(self) -> None:
        "Checks if `.initialize()` was called, before retrieving data."
        assert len(self._gpd_dataframes) > 0, "GPKGMap: Call `.initialize()` before retrieving data!"

    def _assert_layer_available(self, layer: MapLayer) -> None:
        "Checks if layer is available."
        assert layer in self.get_available_map_objects(), f"GPKGMap: MapLayer {layer.name} is unavailable."

    def get_map_metadata(self):
        return self._map_metadata

    def get_available_map_objects(self) -> List[MapLayer]:
        """Inherited, see superclass."""
        self._assert_initialize()
        return list(self._gpd_dataframes.keys())

    def get_map_object(self, object_id: str, layer: MapLayer) -> Optional[AbstractMapObject]:
        """Inherited, see superclass."""

        self._assert_initialize()
        self._assert_layer_available(layer)
        try:
            return self._map_object_getter[layer](object_id)
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} object: {object_id} is unavailable")

    def get_all_map_objects(self, point_2d: Point2D, layer: MapLayer) -> List[AbstractMapObject]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def is_in_layer(self, point: Point2D, layer: MapLayer) -> bool:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_proximal_map_objects(
        self, point_2d: Point2D, radius: float, layers: List[MapLayer]
    ) -> Dict[MapLayer, List[AbstractMapObject]]:
        """Inherited, see superclass."""
        center_point = geom.Point(point_2d.x, point_2d.y)
        patch = center_point.buffer(radius)
        return self.query(geometry=patch, layers=layers, predicate="intersects")

    def query(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]]:
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapLayer, Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]] = defaultdict(
            list
        )
        for layer in layers:
            object_map[layer] = self._query_layer(geometry, layer, predicate, sort, distance)
        return object_map

    def query_object_ids(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[str], Dict[int, List[str]]]]:
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapLayer, Union[List[str], Dict[int, List[str]]]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._query_layer_objects_ids(geometry, layer, predicate, sort, distance)
        return object_map

    def query_nearest(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        return_all: bool = True,
        max_distance: Optional[float] = None,
        return_distance: bool = False,
        exclusive: bool = False,
    ) -> Dict[MapLayer, Union[List[str], Dict[int, List[str]], Dict[int, List[Tuple[str, float]]]]]:
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapLayer, Union[List[str], Dict[int, List[str]]]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._query_layer_nearest(
                geometry, layer, return_all, max_distance, return_distance, exclusive
            )
        return object_map

    def _query_layer(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layer: MapLayer,
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
        layer: MapLayer,
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
        layer: MapLayer,
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
                self._gpd_dataframes[MapLayer.LANE],
                self._gpd_dataframes[MapLayer.LANE_GROUP],
                self._gpd_dataframes[MapLayer.INTERSECTION],
            )
            if id in self._gpd_dataframes[MapLayer.LANE]["id"].tolist()
            else None
        )

    def _get_lane_group(self, id: str) -> Optional[GPKGLaneGroup]:
        return (
            GPKGLaneGroup(
                id,
                self._gpd_dataframes[MapLayer.LANE_GROUP],
                self._gpd_dataframes[MapLayer.LANE],
                self._gpd_dataframes[MapLayer.INTERSECTION],
            )
            if id in self._gpd_dataframes[MapLayer.LANE_GROUP]["id"].tolist()
            else None
        )

    def _get_intersection(self, id: str) -> Optional[GPKGIntersection]:
        return (
            GPKGIntersection(
                id,
                self._gpd_dataframes[MapLayer.INTERSECTION],
                self._gpd_dataframes[MapLayer.LANE],
                self._gpd_dataframes[MapLayer.LANE_GROUP],
            )
            if id in self._gpd_dataframes[MapLayer.INTERSECTION]["id"].tolist()
            else None
        )

    def _get_crosswalk(self, id: str) -> Optional[GPKGCrosswalk]:
        return (
            GPKGCrosswalk(id, self._gpd_dataframes[MapLayer.CROSSWALK])
            if id in self._gpd_dataframes[MapLayer.CROSSWALK]["id"].tolist()
            else None
        )

    def _get_walkway(self, id: str) -> Optional[GPKGWalkway]:
        return (
            GPKGWalkway(id, self._gpd_dataframes[MapLayer.WALKWAY])
            if id in self._gpd_dataframes[MapLayer.WALKWAY]["id"].tolist()
            else None
        )

    def _get_carpark(self, id: str) -> Optional[GPKGCarpark]:
        return (
            GPKGCarpark(id, self._gpd_dataframes[MapLayer.CARPARK])
            if id in self._gpd_dataframes[MapLayer.CARPARK]["id"].tolist()
            else None
        )

    def _get_generic_drivable(self, id: str) -> Optional[GPKGGenericDrivable]:
        return (
            GPKGGenericDrivable(id, self._gpd_dataframes[MapLayer.GENERIC_DRIVABLE])
            if id in self._gpd_dataframes[MapLayer.GENERIC_DRIVABLE]["id"].tolist()
            else None
        )

    def _get_road_edge(self, id: str) -> Optional[GPKGRoadEdge]:
        return (
            GPKGRoadEdge(id, self._gpd_dataframes[MapLayer.ROAD_EDGE])
            if id in self._gpd_dataframes[MapLayer.ROAD_EDGE]["id"].tolist()
            else None
        )

    def _get_road_line(self, id: str) -> Optional[GPKGRoadLine]:
        return (
            GPKGRoadLine(id, self._gpd_dataframes[MapLayer.ROAD_LINE])
            if id in self._gpd_dataframes[MapLayer.ROAD_LINE]["id"].tolist()
            else None
        )

    # def _query_layer(
    #     self,
    #     geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
    #     layer: MapLayer,
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
    #     layer: MapLayer,
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


@lru_cache(maxsize=MAX_LRU_CACHED_TABLES)
def get_global_map_api(dataset: str, location: str) -> GPKGMap:
    PY123D_MAPS_ROOT = Path(os.environ.get("PY123D_MAPS_ROOT"))  # TODO: Remove env variable
    gpkg_path = PY123D_MAPS_ROOT / dataset / f"{dataset}_{location}.gpkg"
    assert gpkg_path.is_file(), f"{dataset}_{location}.gpkg not found in {str(PY123D_MAPS_ROOT)}."
    map_api = GPKGMap(gpkg_path)
    map_api.initialize()
    return map_api


def get_local_map_api(split_name: str, log_name: str) -> GPKGMap:
    PY123D_MAPS_ROOT = Path(os.environ.get("PY123D_MAPS_ROOT"))  # TODO: Remove env variable
    gpkg_path = PY123D_MAPS_ROOT / split_name / f"{log_name}.gpkg"
    assert gpkg_path.is_file(), f"{log_name}.gpkg not found in {str(PY123D_MAPS_ROOT)}."
    map_api = GPKGMap(gpkg_path)
    map_api.initialize()
    return map_api
