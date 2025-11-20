import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import shapely.geometry as geom

from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.map_writer.utils.gpkg_utils import IntIDMapping
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.map_objects.map_objects import (
    BaseMapLineObject,
    BaseMapSurfaceObject,
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    RoadEdge,
    RoadLine,
    StopZone,
    Walkway,
)
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.geometry.polyline import Polyline3D

MAP_OBJECT_DATA = Dict[str, List[Union[str, int, float, bool, geom.base.BaseGeometry]]]

logging.getLogger("pyogrio._io").disabled = True


class GPKGMapWriter(AbstractMapWriter):
    """Abstract base class for map writers."""

    def __init__(self, maps_root: Union[str, Path], remap_ids: bool = False) -> None:
        self._maps_root = Path(maps_root)
        self._crs: str = "EPSG:4326"  # WGS84
        self._remap_ids = remap_ids

        # Data to be written to the map for each object type
        self._map_data: Optional[Dict[MapLayer, MAP_OBJECT_DATA]] = None
        self._map_file: Optional[Path] = None
        self._map_metadata: Optional[MapMetadata] = None

    def reset(self, dataset_converter_config: DatasetConverterConfig, map_metadata: MapMetadata) -> bool:
        """Inherited, see superclass."""

        map_needs_writing: bool = False

        if dataset_converter_config.include_map:
            if map_metadata.map_is_local:
                split, log_name = map_metadata.split, map_metadata.log_name
                map_file = self._maps_root / split / f"{log_name}.gpkg"
            else:
                dataset, location = map_metadata.dataset, map_metadata.location
                map_file = self._maps_root / dataset / f"{dataset}_{location}.gpkg"

            map_needs_writing = dataset_converter_config.force_map_conversion or not map_file.exists()
            if map_needs_writing:
                # Reset all map layers and update map file / metadata
                self._map_data = {map_layer: defaultdict(list) for map_layer in MapLayer}
                self._map_file = map_file
                self._map_metadata = map_metadata

        return map_needs_writing

    def write_lane(self, lane: Lane) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.LANE, lane)
        self._map_data[MapLayer.LANE]["lane_group_id"].append(lane.lane_group_id)
        self._map_data[MapLayer.LANE]["left_boundary"].append(lane.left_boundary.linestring)
        self._map_data[MapLayer.LANE]["right_boundary"].append(lane.right_boundary.linestring)
        self._map_data[MapLayer.LANE]["centerline"].append(lane.centerline.linestring)
        self._map_data[MapLayer.LANE]["left_lane_id"].append(lane.left_lane_id)
        self._map_data[MapLayer.LANE]["right_lane_id"].append(lane.right_lane_id)
        self._map_data[MapLayer.LANE]["predecessor_ids"].append(lane.predecessor_ids)
        self._map_data[MapLayer.LANE]["successor_ids"].append(lane.successor_ids)
        self._map_data[MapLayer.LANE]["speed_limit_mps"].append(lane.speed_limit_mps)

    def write_lane_group(self, lane_group: LaneGroup) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.LANE_GROUP, lane_group)
        self._map_data[MapLayer.LANE_GROUP]["lane_ids"].append(lane_group.lane_ids)
        self._map_data[MapLayer.LANE_GROUP]["intersection_id"].append(lane_group.intersection_id)
        self._map_data[MapLayer.LANE_GROUP]["predecessor_ids"].append(lane_group.predecessor_ids)
        self._map_data[MapLayer.LANE_GROUP]["successor_ids"].append(lane_group.successor_ids)
        self._map_data[MapLayer.LANE_GROUP]["left_boundary"].append(lane_group.left_boundary.linestring)
        self._map_data[MapLayer.LANE_GROUP]["right_boundary"].append(lane_group.right_boundary.linestring)

    def write_intersection(self, intersection: Intersection) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.INTERSECTION, intersection)
        self._map_data[MapLayer.INTERSECTION]["lane_group_ids"].append(intersection.lane_group_ids)

    def write_crosswalk(self, crosswalk: Crosswalk) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.CROSSWALK, crosswalk)

    def write_carpark(self, carpark: Carpark) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.CARPARK, carpark)

    def write_walkway(self, walkway: Walkway) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.WALKWAY, walkway)

    def write_generic_drivable(self, obj: GenericDrivable) -> None:
        """Inherited, see superclass."""
        self._write_surface_layer(MapLayer.GENERIC_DRIVABLE, obj)

    def write_stop_zone(self, stop_zone: StopZone) -> None:
        """Inherited, see superclass."""
        # self._write_line_layer(MapLayer.STOP_LINE, stop_line)
        raise NotImplementedError("Stop zones are not yet supported in GPKG maps.")

    def write_road_edge(self, road_edge: RoadEdge) -> None:
        """Inherited, see superclass."""
        self._write_line_layer(MapLayer.ROAD_EDGE, road_edge)
        self._map_data[MapLayer.ROAD_EDGE]["road_edge_type"].append(int(road_edge.road_edge_type))

    def write_road_line(self, road_line: RoadLine) -> None:
        """Inherited, see superclass."""
        self._write_line_layer(MapLayer.ROAD_LINE, road_line)
        self._map_data[MapLayer.ROAD_LINE]["road_line_type"].append(int(road_line.road_line_type))

    def close(self) -> None:
        """Inherited, see superclass."""

        if self._map_file is not None or self._map_data is not None:
            if not self._map_file.parent.exists():
                self._map_file.parent.mkdir(parents=True, exist_ok=True)

            # Accumulate GeoDataFrames for each map layer
            map_gdf: Dict[MapLayer, gpd.GeoDataFrame] = {}
            for map_layer, layer_data in self._map_data.items():
                if len(layer_data["id"]) > 0:
                    df = pd.DataFrame(layer_data)
                    map_gdf[map_layer] = gpd.GeoDataFrame(df, geometry="geometry", crs=self._crs)
                else:
                    map_gdf[map_layer] = gpd.GeoDataFrame(
                        {"id": [], "geometry": []}, geometry="geometry", crs=self._crs
                    )

            # Optionally remap string IDs to integers
            if self._remap_ids:
                _map_ids_to_integer(map_gdf)

            # Write each map layer to the GPKG file
            for map_layer, gdf in map_gdf.items():
                gdf.to_file(self._map_file, driver="GPKG", layer=map_layer.serialize())

            # Write map metadata as a separate layer
            metadata_df = gpd.GeoDataFrame(pd.DataFrame([self._map_metadata.to_dict()]))
            metadata_df.to_file(self._map_file, driver="GPKG", layer="map_metadata")

        del self._map_file, self._map_data, self._map_metadata
        self._map_file = None
        self._map_data = None
        self._map_metadata = None

    def _assert_initialized(self) -> None:
        assert self._map_data is not None, "Call reset() before writing data."
        assert self._map_file is not None, "Call reset() before writing data."
        assert self._map_metadata is not None, "Call reset() before writing data."

    def _write_surface_layer(self, layer: MapLayer, surface_object: BaseMapSurfaceObject) -> None:
        """Helper to write surface map objects.

        :param layer: map layer of surface object
        :param surface_object: surface map object to write
        """
        self._assert_initialized()
        self._map_data[layer]["id"].append(surface_object.object_id)
        # NOTE: if the outline has a z-coordinate, we store it, otherwise we infer from the outline from the polygon
        if isinstance(surface_object.outline, Polyline3D):
            self._map_data[layer]["outline"].append(surface_object.outline.linestring)
        self._map_data[layer]["geometry"].append(surface_object.shapely_polygon)

    def _write_line_layer(self, layer: MapLayer, line_object: BaseMapLineObject) -> None:
        """Helper to write line map objects.

        :param layer: map layer of line object
        :param line_object: line map object to write
        """
        self._assert_initialized()
        self._map_data[layer]["id"].append(line_object.object_id)
        self._map_data[layer]["geometry"].append(line_object.shapely_linestring)


def _map_ids_to_integer(map_dfs: Dict[MapLayer, gpd.GeoDataFrame]) -> None:
    """Helper function to remap string IDs to integers in the map dataframes."""

    # initialize id mappings
    lane_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.LANE]["id"])
    lane_group_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.LANE_GROUP]["id"])
    intersection_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.INTERSECTION]["id"])

    walkway_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.WALKWAY]["id"])
    carpark_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.CARPARK]["id"])
    generic_drivable_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.GENERIC_DRIVABLE]["id"])
    road_line_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.ROAD_LINE]["id"])
    road_edge_id_mapping = IntIDMapping.from_series(map_dfs[MapLayer.ROAD_EDGE]["id"])

    # 1. Remap lane ids in LANE layer
    if len(map_dfs[MapLayer.LANE]) > 0:
        map_dfs[MapLayer.LANE]["id"] = map_dfs[MapLayer.LANE]["id"].apply(lambda x: lane_id_mapping.map(x))
        map_dfs[MapLayer.LANE]["lane_group_id"] = map_dfs[MapLayer.LANE]["lane_group_id"].apply(
            lambda x: lane_group_id_mapping.map(x)
        )
        for column in ["predecessor_ids", "successor_ids"]:
            map_dfs[MapLayer.LANE][column] = map_dfs[MapLayer.LANE][column].apply(lambda x: lane_id_mapping.map_list(x))
        for column in ["left_lane_id", "right_lane_id"]:
            map_dfs[MapLayer.LANE][column] = map_dfs[MapLayer.LANE][column].apply(lambda x: lane_id_mapping.map(x))

    # 2. Remap lane group ids in LANE_GROUP
    if len(map_dfs[MapLayer.LANE_GROUP]) > 0:
        map_dfs[MapLayer.LANE_GROUP]["id"] = map_dfs[MapLayer.LANE_GROUP]["id"].apply(
            lambda x: lane_group_id_mapping.map(x)
        )
        map_dfs[MapLayer.LANE_GROUP]["lane_ids"] = map_dfs[MapLayer.LANE_GROUP]["lane_ids"].apply(
            lambda x: lane_id_mapping.map_list(x)
        )
        map_dfs[MapLayer.LANE_GROUP]["intersection_id"] = map_dfs[MapLayer.LANE_GROUP]["intersection_id"].apply(
            lambda x: intersection_id_mapping.map(x)
        )
        for column in ["predecessor_ids", "successor_ids"]:
            map_dfs[MapLayer.LANE_GROUP][column] = map_dfs[MapLayer.LANE_GROUP][column].apply(
                lambda x: lane_group_id_mapping.map_list(x)
            )

    # 3. Remap lane group ids in INTERSECTION
    if len(map_dfs[MapLayer.INTERSECTION]) > 0:
        map_dfs[MapLayer.INTERSECTION]["id"] = map_dfs[MapLayer.INTERSECTION]["id"].apply(
            lambda x: intersection_id_mapping.map(x)
        )
        map_dfs[MapLayer.INTERSECTION]["lane_group_ids"] = map_dfs[MapLayer.INTERSECTION]["lane_group_ids"].apply(
            lambda x: lane_group_id_mapping.map_list(x)
        )

    # 4. Remap ids in other layers
    if len(map_dfs[MapLayer.WALKWAY]) > 0:
        map_dfs[MapLayer.WALKWAY]["id"] = map_dfs[MapLayer.WALKWAY]["id"].apply(lambda x: walkway_id_mapping.map(x))
    if len(map_dfs[MapLayer.CARPARK]) > 0:
        map_dfs[MapLayer.CARPARK]["id"] = map_dfs[MapLayer.CARPARK]["id"].apply(lambda x: carpark_id_mapping.map(x))
    if len(map_dfs[MapLayer.GENERIC_DRIVABLE]) > 0:
        map_dfs[MapLayer.GENERIC_DRIVABLE]["id"] = map_dfs[MapLayer.GENERIC_DRIVABLE]["id"].apply(
            lambda x: generic_drivable_id_mapping.map(x)
        )
    if len(map_dfs[MapLayer.ROAD_LINE]) > 0:
        map_dfs[MapLayer.ROAD_LINE]["id"] = map_dfs[MapLayer.ROAD_LINE]["id"].apply(
            lambda x: road_line_id_mapping.map(x)
        )
    if len(map_dfs[MapLayer.ROAD_EDGE]) > 0:
        map_dfs[MapLayer.ROAD_EDGE]["id"] = map_dfs[MapLayer.ROAD_EDGE]["id"].apply(
            lambda x: road_edge_id_mapping.map(x)
        )
