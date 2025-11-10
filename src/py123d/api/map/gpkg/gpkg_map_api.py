from __future__ import annotations

import ast
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Final, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import shapely
import shapely.geometry as geom

from py123d.api.map.gpkg.gpkg_utils import get_row_with_value, load_gdf_with_geometry_columns
from py123d.api.map.map_api import MapAPI
from py123d.datatypes.map_objects.base_map_objects import BaseMapObject
from py123d.datatypes.map_objects.map_layer_types import MapLayer, RoadEdgeType, RoadLineType
from py123d.datatypes.map_objects.map_objects import (
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    RoadEdge,
    RoadLine,
    Walkway,
)
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.geometry import Point2D
from py123d.geometry.polyline import Polyline3D
from py123d.script.utils.dataset_path_utils import get_dataset_paths

USE_ARROW: bool = True
MAX_LRU_CACHED_TABLES: Final[int] = 128  # TODO: add to some configs


class GPKGMapAPI(MapAPI):
    def __init__(self, file_path: Path) -> None:

        self._file_path = file_path
        self._map_object_getter: Dict[MapLayer, Callable[[str], Optional[BaseMapObject]]] = {
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

    def _initialize(self) -> None:
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
        assert layer in self.get_available_map_layers(), f"GPKGMap: MapLayer {layer.name} is unavailable."

    def get_map_metadata(self):
        """Inherited, see superclass."""
        return self._map_metadata

    def get_available_map_layers(self) -> List[MapLayer]:
        """Inherited, see superclass."""
        self._assert_initialize()
        return list(self._gpd_dataframes.keys())

    def get_map_object(self, object_id: str, layer: MapLayer) -> Optional[BaseMapObject]:
        """Inherited, see superclass."""

        self._assert_initialize()
        self._assert_layer_available(layer)
        try:
            return self._map_object_getter[layer](object_id)
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} object: {object_id} is unavailable")

    def get_all_map_objects(self, point_2d: Point2D, layer: MapLayer) -> List[BaseMapObject]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def is_in_layer(self, point: Point2D, layer: MapLayer) -> bool:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_proximal_map_objects(
        self, point_2d: Point2D, radius: float, layers: List[MapLayer]
    ) -> Dict[MapLayer, List[BaseMapObject]]:
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
    ) -> Dict[MapLayer, Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]]:
        """Inherited, see superclass."""
        supported_layers = self.get_available_map_layers()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"
        object_map: Dict[MapLayer, Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]] = defaultdict(list)
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
        """Inherited, see superclass."""
        supported_layers = self.get_available_map_layers()
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
        """Inherited, see superclass."""
        supported_layers = self.get_available_map_layers()
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
    ) -> Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]:
        """Helper method to query a single layer."""

        queried_indices = self._gpd_dataframes[layer].sindex.query(
            geometry, predicate=predicate, sort=sort, distance=distance
        )

        if queried_indices.ndim == 2:
            query_dict: Dict[int, List[BaseMapObject]] = defaultdict(list)
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
        """Helper method to query a single layer."""

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
        """Helper method to query a single layer."""

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

    def _get_lane(self, id: str) -> Optional[Lane]:
        """Helper method for getting a lane by its ID."""
        lane: Optional[Lane] = None
        lane_row = get_row_with_value(self._gpd_dataframes[MapLayer.LANE], "id", id)

        if lane_row is not None:
            object_id: str = lane_row["id"]
            lane_group_id: str = lane_row["lane_group_id"]
            left_boundary: Polyline3D = Polyline3D.from_linestring(lane_row["left_boundary"])
            right_boundary: Optional[Polyline3D] = Polyline3D.from_linestring(lane_row["right_boundary"])
            centerline: Polyline3D = Polyline3D.from_linestring(lane_row["centerline"])
            left_lane_id: Optional[str] = lane_row["left_lane_id"]
            right_lane_id: Optional[str] = lane_row["right_lane_id"]
            predecessor_ids: List[str] = ast.literal_eval(lane_row.predecessor_ids)
            successor_ids: List[str] = ast.literal_eval(lane_row.successor_ids)
            speed_limit_mps: Optional[float] = lane_row["speed_limit_mps"]
            outline: Optional[Polyline3D] = (
                Polyline3D.from_linestring(lane_row["outline"]) if lane_row["outline"] is not None else None
            )
            geometry: geom.LineString = lane_row["geometry"]

            lane = Lane(
                object_id=object_id,
                lane_group_id=lane_group_id,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                centerline=centerline,
                left_lane_id=left_lane_id,
                right_lane_id=right_lane_id,
                predecessor_ids=predecessor_ids,
                successor_ids=successor_ids,
                speed_limit_mps=speed_limit_mps,
                outline=outline,
                geometry=geometry,
                map_api=self,
            )

        return lane

    def _get_lane_group(self, id: str) -> Optional[LaneGroup]:
        """Helper method for getting a lane group by its ID."""
        lane_group: Optional[LaneGroup] = None
        lane_group_row = get_row_with_value(self._gpd_dataframes[MapLayer.LANE_GROUP], "id", id)
        if lane_group_row is not None:

            object_id: str = lane_group_row["id"]
            lane_ids: List[str] = ast.literal_eval(lane_group_row.lane_ids)
            left_boundary: Polyline3D = Polyline3D.from_linestring(lane_group_row["left_boundary"])
            right_boundary: Optional[Polyline3D] = Polyline3D.from_linestring(lane_group_row["right_boundary"])
            intersection_id: Optional[str] = lane_group_row["intersection_id"]
            predecessor_ids: Optional[List[str]] = ast.literal_eval(lane_group_row.predecessor_ids)
            successor_ids: Optional[List[str]] = ast.literal_eval(lane_group_row.successor_ids)
            outline: Optional[Polyline3D] = (
                Polyline3D.from_linestring(lane_group_row["outline"]) if lane_group_row["outline"] is not None else None
            )
            geometry: geom.Polygon = lane_group_row["geometry"]

            lane_group = LaneGroup(
                object_id=object_id,
                lane_ids=lane_ids,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                intersection_id=intersection_id,
                predecessor_ids=predecessor_ids,
                successor_ids=successor_ids,
                outline=outline,
                geometry=geometry,
                map_api=self,
            )

        return lane_group

    def _get_intersection(self, id: str) -> Optional[Intersection]:
        """Helper method for getting an intersection by its ID."""

        intersection: Optional[Intersection] = None
        intersection_row = get_row_with_value(self._gpd_dataframes[MapLayer.INTERSECTION], "id", id)
        if intersection_row is not None:

            object_id: str = intersection_row["id"]
            lane_group_ids: List[str] = ast.literal_eval(intersection_row.lane_group_ids)
            outline: Optional[Polyline3D] = (
                Polyline3D.from_linestring(intersection_row["outline"])
                if intersection_row["outline"] is not None
                else None
            )
            geometry: geom.Polygon = intersection_row["geometry"]

            intersection = Intersection(
                object_id=object_id,
                lane_group_ids=lane_group_ids,
                outline=outline,
                geometry=geometry,
                map_api=self,
            )

        return intersection

    def _get_crosswalk(self, id: str) -> Optional[Crosswalk]:
        """Helper method for getting a crosswalk by its ID."""

        crosswalk: Optional[Crosswalk] = None
        crosswalk_row = get_row_with_value(self._gpd_dataframes[MapLayer.CROSSWALK], "id", id)
        if crosswalk_row is not None:

            object_id: str = crosswalk_row["id"]
            outline: Polyline3D = Polyline3D.from_linestring(crosswalk_row["outline"])
            geometry: geom.Polygon = crosswalk_row["geometry"]

            crosswalk = Crosswalk(
                object_id=object_id,
                outline=outline,
                geometry=geometry,
            )

        return crosswalk

    def _get_walkway(self, id: str) -> Optional[Walkway]:
        """Helper method for getting a walkway by its ID."""

        walkway: Optional[Walkway] = None
        walkway_row = get_row_with_value(self._gpd_dataframes[MapLayer.WALKWAY], "id", id)
        if walkway_row is not None:

            object_id: str = walkway_row["id"]
            outline: Polyline3D = Polyline3D.from_linestring(walkway_row["outline"])
            geometry: geom.Polygon = walkway_row["geometry"]

            walkway = Walkway(
                object_id=object_id,
                outline=outline,
                geometry=geometry,
            )

        return walkway

    def _get_carpark(self, id: str) -> Optional[Carpark]:
        """Helper method for getting a carpark by its ID."""

        carpark: Optional[Carpark] = None
        carpark_row = get_row_with_value(self._gpd_dataframes[MapLayer.CARPARK], "id", id)
        if carpark_row is not None:

            object_id: str = carpark_row["id"]
            outline: Polyline3D = Polyline3D.from_linestring(carpark_row["outline"])
            geometry: geom.Polygon = carpark_row["geometry"]

            carpark = Carpark(
                object_id=object_id,
                outline=outline,
                geometry=geometry,
            )

        return carpark

    def _get_generic_drivable(self, id: str) -> Optional[GenericDrivable]:
        """Helper method for getting a generic drivable area by its ID."""

        generic_drivable: Optional[GenericDrivable] = None
        generic_drivable_row = get_row_with_value(self._gpd_dataframes[MapLayer.GENERIC_DRIVABLE], "id", id)
        if generic_drivable_row is not None:

            object_id: str = generic_drivable_row["id"]
            outline: Polyline3D = Polyline3D.from_linestring(generic_drivable_row["outline"])
            geometry: geom.Polygon = generic_drivable_row["geometry"]

            generic_drivable = GenericDrivable(
                object_id=object_id,
                outline=outline,
                geometry=geometry,
            )

        return generic_drivable

    def _get_road_edge(self, id: str) -> Optional[RoadEdge]:
        """Helper method for getting a road edge by its ID."""

        road_edge: Optional[RoadEdge] = None
        road_edge_row = get_row_with_value(self._gpd_dataframes[MapLayer.ROAD_EDGE], "id", id)
        if road_edge_row is not None:

            object_id: str = road_edge_row["id"]
            polyline: Polyline3D = Polyline3D.from_linestring(road_edge_row["geometry"])
            road_edge_type: RoadEdgeType = RoadEdgeType(road_edge_row["road_edge_type"])

            road_edge = RoadEdge(
                object_id=object_id,
                road_edge_type=road_edge_type,
                polyline=polyline,
            )

        return road_edge

    def _get_road_line(self, id: str) -> Optional[RoadLine]:
        """Helper method for getting a road line by its ID."""

        road_line: Optional[RoadLine] = None
        road_line_row = get_row_with_value(self._gpd_dataframes[MapLayer.ROAD_LINE], "id", id)
        if road_line_row is not None:

            object_id: str = road_line_row["id"]
            polyline: Polyline3D = Polyline3D.from_linestring(road_line_row["geometry"])
            road_line_type: RoadLineType = RoadLineType(road_line_row["road_line_type"])

            road_line = RoadLine(
                object_id=object_id,
                road_line_type=road_line_type,
                polyline=polyline,
            )

        return road_line


@lru_cache(maxsize=MAX_LRU_CACHED_TABLES)
def get_global_map_api(dataset: str, location: str) -> GPKGMapAPI:
    PY123D_MAPS_ROOT: Path = Path(get_dataset_paths().py123d_maps_root)
    gpkg_path = PY123D_MAPS_ROOT / dataset / f"{dataset}_{location}.gpkg"
    assert gpkg_path.is_file(), f"{dataset}_{location}.gpkg not found in {str(PY123D_MAPS_ROOT)}."
    map_api = GPKGMapAPI(gpkg_path)
    map_api._initialize()
    return map_api


def get_local_map_api(split_name: str, log_name: str) -> GPKGMapAPI:
    PY123D_MAPS_ROOT: Path = Path(get_dataset_paths().py123d_maps_root)
    gpkg_path = PY123D_MAPS_ROOT / split_name / f"{log_name}.gpkg"
    assert gpkg_path.is_file(), f"{log_name}.gpkg not found in {str(PY123D_MAPS_ROOT)}."
    map_api = GPKGMapAPI(gpkg_path)
    map_api._initialize()
    return map_api
