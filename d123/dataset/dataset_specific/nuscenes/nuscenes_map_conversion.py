import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.validation import make_valid
from d123.dataset.maps.map_datatypes import MapLayer, RoadLineType
from nuscenes.map_expansion.arcline_path_utils import discretize_lane
import shapely

MapSurfaceType = MapLayer

NUSCENES_MAPS = {
    "boston-seaport": "boston-seaport",
    "singapore-hollandvillage": "singapore-hollandvillage",
    "singapore-onenorth": "singapore-onenorth",
    "singapore-queenstown": "singapore-queenstown"
}
NUSCENES_MAPS_ROOT = os.environ["NUSCENES_MAPS_ROOT"]


def flip_linestring(linestring: LineString) -> LineString:
    return LineString(linestring.coords[::-1])


def lines_same_direction(centerline: LineString, boundary: LineString) -> bool:
    center_start = np.array(centerline.coords[0])
    center_end = np.array(centerline.coords[-1])
    boundary_start = np.array(boundary.coords[0])
    boundary_end = np.array(boundary.coords[-1])

    same_dir_dist = np.linalg.norm(center_start - boundary_start) + np.linalg.norm(center_end - boundary_end)
    opposite_dir_dist = np.linalg.norm(center_start - boundary_end) + np.linalg.norm(center_end - boundary_start)

    return same_dir_dist <= opposite_dir_dist


def align_boundary_direction(centerline: LineString, boundary: LineString) -> LineString:
    if not lines_same_direction(centerline, boundary):
        return flip_linestring(boundary)
    return boundary


class NuscenesMapConverter:
    def __init__(self, map_path: Path) -> None:
        self.dataroot = NUSCENES_MAPS_ROOT
        self.map_path = map_path
        self.nusc_maps: Dict[str, NuScenesMap] = {}
        self._gdf: Dict[str, gpd.GeoDataFrame] = {}

        # Initialize all nuscenes maps
        for map_name, map_key in NUSCENES_MAPS.items():
            self.nusc_maps[map_name] = NuScenesMap(dataroot=NUSCENES_MAPS_ROOT, map_name=map_key)

    def convert(self, map_name: str) -> None:
        assert map_name in NUSCENES_MAPS, f"Map name {map_name} is not supported."

        # Load all layers from nuscenes map
        self._load_dataframes(map_name)

        # Extract and process each layer
        lane_df = self._extract_lane_dataframe(map_name)
        lane_group_df = self._extract_lane_group_dataframe(map_name)
        intersection_df = self._extract_intersection_dataframe()
        crosswalk_df = self._extract_crosswalk_dataframe()
        walkway_df = self._extract_walkway_dataframe()
        carpark_df = self._extract_carpark_dataframe()
        generic_drivable_df = self._extract_generic_drivable_dataframe()
        stop_line_df = self._extract_stop_line_dataframe()
        road_line_df = self._extract_road_line_dataframe()

        def ensure_nonempty(gdf: gpd.GeoDataFrame, cols: list) -> gpd.GeoDataFrame:
            if gdf is None or gdf.empty:
                col_dict = {col: [] for col in cols}
                col_dict["geometry"] = []
                return gpd.GeoDataFrame(col_dict, geometry="geometry")
            return gdf

        lane_df = ensure_nonempty(lane_df, ["id", "lane_group_id", "speed_limit_mps",
                                            "predecessor_ids", "successor_ids",
                                            "left_boundary", "right_boundary", "baseline_path"])
        lane_group_df = ensure_nonempty(lane_group_df, ["id", "lane_ids", "is_intersection",
                                                        "predecessor_lane_group_ids", "successor_lane_group_ids",
                                                        "left_boundary", "right_boundary"])
        intersection_df = ensure_nonempty(intersection_df, ["id", "lane_group_ids"])
        crosswalk_df = ensure_nonempty(crosswalk_df, ["id"])
        walkway_df = ensure_nonempty(walkway_df, ["id"])
        carpark_df = ensure_nonempty(carpark_df, ["id"])
        generic_drivable_df = ensure_nonempty(generic_drivable_df, ["id"])
        stop_line_df = ensure_nonempty(stop_line_df, ["id"])
        road_line_df = ensure_nonempty(road_line_df, ["id"])

        # Create output directory if needed
        if not self.map_path.exists():
            self.map_path.mkdir(parents=True, exist_ok=True)

        # Save to GeoPackage
        map_file_name = self.map_path / f"nuscenes_{map_name}.gpkg"
        lane_df.to_file(map_file_name, layer=MapSurfaceType.LANE.serialize(), driver="GPKG", mode="w")
        lane_group_df.to_file(map_file_name, layer=MapSurfaceType.LANE_GROUP.serialize(), driver="GPKG", mode="a")
        intersection_df.to_file(map_file_name, layer=MapSurfaceType.INTERSECTION.serialize(), driver="GPKG", mode="a")
        crosswalk_df.to_file(map_file_name, layer=MapSurfaceType.CROSSWALK.serialize(), driver="GPKG", mode="a")
        walkway_df.to_file(map_file_name, layer=MapSurfaceType.WALKWAY.serialize(), driver="GPKG", mode="a")
        carpark_df.to_file(map_file_name, layer=MapSurfaceType.CARPARK.serialize(), driver="GPKG", mode="a")
        generic_drivable_df.to_file(
            map_file_name, layer=MapSurfaceType.GENERIC_DRIVABLE.serialize(), driver="GPKG", mode="a")
        stop_line_df.to_file(map_file_name, layer=MapSurfaceType.STOP_LINE.serialize(), driver="GPKG", mode="a")
        road_line_df.to_file(map_file_name, layer=MapSurfaceType.ROAD_LINE.serialize(), driver="GPKG", mode="a")

    def _load_dataframes(self, map_name: str) -> None:
        """Load all relevant layers from nuscenes map into GeoDataFrames."""
        nusc_map = self.nusc_maps[map_name]
        self._gdf = {}

        # Define layer processing
        layers = {
            "lane": ("lane", self._process_polygon_layer),
            "road_segment": ("road_segment", self._process_polygon_layer),
            "intersection": ("road_block", self._process_polygon_layer),
            "ped_crossing": ("ped_crossing", self._process_polygon_layer),
            "walkway": ("walkway", self._process_polygon_layer),
            "carpark": ("carpark_area", self._process_polygon_layer),
            "road": ("drivable_area", self._process_polygon_layer),
            "road_divider": ("road_divider", self._process_line_layer_with_type),
            "lane_divider": ("lane_divider", self._process_line_layer_with_type), 
            "stop_line": ("stop_line", self._process_polygon_layer),
        }

        for layer_key, (table_name, processor) in layers.items():
            records = getattr(nusc_map, table_name)

            if processor == self._process_line_layer_with_type:
                geoms, ids, line_types = processor(records, nusc_map)
                if geoms:
                    gdf = gpd.GeoDataFrame({"fid": ids, "line_type": line_types}, geometry=geoms)
                    self._gdf[layer_key] = gdf
            else:
                geoms, ids = processor(records, nusc_map)
                if geoms:
                    gdf = gpd.GeoDataFrame({"fid": ids}, geometry=geoms)
                    self._gdf[layer_key] = gdf

        road_divider_gdf = self._gdf.get("road_divider", gpd.GeoDataFrame())
        lane_divider_gdf = self._gdf.get("lane_divider", gpd.GeoDataFrame())

        if not road_divider_gdf.empty or not lane_divider_gdf.empty:
            self._gdf["road_line"] = gpd.GeoDataFrame(
                pd.concat([road_divider_gdf, lane_divider_gdf], ignore_index=True)
            )

    def _process_polygon_layer(self, records, nusc_map) -> Tuple[List, List]:
        geoms, ids = [], []
        for record in records:
            try:
                if "polygon_token" in record:
                    polygon = nusc_map.extract_polygon(record["polygon_token"])
                elif "polygon_tokens" in record:
                    polygons = []
                    for p_token in record["polygon_tokens"]:
                        poly = nusc_map.extract_polygon(p_token)
                        if poly.is_valid:
                            polygons.append(poly)
                    if not polygons:
                        continue  
                    polygon = MultiPolygon(polygons)                 
                else:
                    continue
                if not polygon.is_valid:
                    continue

                if polygon.geom_type == "Polygon":
                    geoms.append(polygon)
                    ids.append(record["token"])
                elif polygon.geom_type == "MultiPolygon":
                    geoms.extend(polygon.geoms)
                    ids.extend([record["token"]] * len(polygon.geoms))

            except Exception:
                continue

        return geoms, ids

    def _process_line_layer_with_type(self, records, nusc_map) -> Tuple[List, List, List]:
        """Process line layers into geometries, IDs and line types."""
        geoms, ids, line_types = [], [], []

        nuscenes_to_road_line_type = {
            "SINGLE_SOLID_WHITE": RoadLineType.SOLID_SINGLE_WHITE,
            "DOUBLE_DASHED_WHITE": RoadLineType.BROKEN_SINGLE_WHITE,  
            "SINGLE_SOLID_YELLOW": RoadLineType.SOLID_SINGLE_YELLOW,
        }

        line_token_to_type = {}

        for lane_record in nusc_map.lane:
            for seg in lane_record.get("left_lane_divider_segments", []):
                line_token = seg.get("line_token")
                seg_type = seg.get("segment_type")
                if line_token and seg_type:
                    line_token_to_type[line_token] = seg_type

            for seg in lane_record.get("right_lane_divider_segments", []):
                line_token = seg.get("line_token")
                seg_type = seg.get("segment_type")
                if line_token and seg_type:
                    line_token_to_type[line_token] = seg_type

        for record in records:
            line = nusc_map.extract_line(record["line_token"])
            if line.is_valid:
                geoms.append(line)
                ids.append(record["token"])

                nuscenes_line_type = line_token_to_type.get(record["line_token"], "UNKNOWN")
                line_type = nuscenes_to_road_line_type.get(nuscenes_line_type, RoadLineType.UNKNOWN)
                line_types.append(line_type)

        return geoms, ids, line_types

    def _process_line_layer(self, records, nusc_map) -> Tuple[List, List]:
        """Process line layers into geometries and IDs."""
        geoms, ids = [], []
        for record in records:
            line = nusc_map.extract_line(record["line_token"])
            if line.is_valid:
                geoms.append(line)
                ids.append(record["token"])
        return geoms, ids

    def _extract_lane_dataframe(self, map_name: str) -> gpd.GeoDataFrame:
        """Create lane GeoDataFrame with topology information (基于 arcline_path_3)."""
        if "lane" not in self._gdf:
            return gpd.GeoDataFrame()

        nusc_map = self.nusc_maps[map_name]
        lane_gdf = self._gdf["lane"]
        lane_records = nusc_map.lane

        lane_data = []
        for _, row in lane_gdf.iterrows():
            token = row["fid"]
            lane_record = next((lr for lr in lane_records if lr["token"] == token), None)
            if not lane_record:
                continue

            incoming = nusc_map.get_incoming_lane_ids(token)
            outgoing = nusc_map.get_outgoing_lane_ids(token)

            lane_connectors = []
            for connector in nusc_map.lane_connector:
                if connector.get("incoming_lane") == token or connector.get("outgoing_lane") == token:
                    lane_connectors.append(connector["token"])

            left_boundary = self._get_lane_boundary(token, "left", nusc_map)
            right_boundary = self._get_lane_boundary(token, "right", nusc_map)

            baseline_path: Optional[shapely.geometry.LineString] = None
            if token in nusc_map.arcline_path_3:
                arc_path = nusc_map.arcline_path_3[token]
                try:
                    points = discretize_lane(arc_path, resolution_meters=0.1)
                    xy_points = [(p[0], p[1]) for p in points]
                    baseline_path = shapely.geometry.LineString(xy_points)
                except Exception:
                    continue

            if baseline_path and left_boundary:
                left_boundary = align_boundary_direction(baseline_path, left_boundary)
            if baseline_path and right_boundary:
                right_boundary = align_boundary_direction(baseline_path, right_boundary)

            lane_data.append({
                "id": token,
                "lane_group_id": lane_record.get("road_segment_token", None),
                "speed_limit_mps": 0.0,
                "predecessor_ids": incoming,
                "successor_ids": outgoing,
                "lane_connector_ids": lane_connectors, 
                "left_boundary": left_boundary,
                "right_boundary": right_boundary,
                "baseline_path": baseline_path,
                "geometry": row.geometry
            })

        return gpd.GeoDataFrame(lane_data)

    def _get_lane_boundary(self, lane_token: str, side: str, nusc_map: NuScenesMap) -> Optional[LineString]:
        """Extract lane boundary geometry."""
        cache_key = f"{lane_token}_{side}"
        if hasattr(self, '_boundary_cache') and cache_key in self._boundary_cache:
            return self._boundary_cache[cache_key]

        lane_record = next((lr for lr in nusc_map.lane if lr["token"] == lane_token), None)
        if not lane_record:
            return None

        divider_segment_nodes_key = f"{side}_lane_divider_segment_nodes"

        if divider_segment_nodes_key in lane_record and lane_record[divider_segment_nodes_key]:
            nodes = lane_record[divider_segment_nodes_key]
            boundary = LineString([(node['x'], node['y']) for node in nodes])
            
            if not hasattr(self, '_boundary_cache'):
                self._boundary_cache = {}
            self._boundary_cache[cache_key] = boundary
            return boundary

        return None

    def _extract_lane_group_dataframe(self, map_name: str) -> gpd.GeoDataFrame:
        """Create lane group GeoDataFrame with topology information."""
        if "road_segment" not in self._gdf:
            return gpd.GeoDataFrame()

        nusc_map = self.nusc_maps[map_name]
        road_segments = nusc_map.road_segment
        lane_gdf = self._gdf["lane"]
        lane_group_gdf = self._gdf["road_segment"]

        lane_group_data = []
        for _, row in lane_group_gdf.iterrows():
            token = row["fid"]
            segment = next((rs for rs in road_segments if rs["token"] == token), None)
            if not segment:
                continue

            candidate_lanes = lane_gdf[lane_gdf.intersects(row.geometry)]
            lane_ids = candidate_lanes["fid"].tolist()

            incoming, outgoing = self._get_connected_segments(token, nusc_map)

            baseline_path = None
            if not candidate_lanes.empty:
                first_lane_token = candidate_lanes.iloc[0]["fid"]
                first_lane = next((lr for lr in nusc_map.lane if lr["token"] == first_lane_token), None)
                if first_lane and "path" in first_lane and first_lane["path"]:
                    baseline_path = LineString(first_lane["path"])

            left_boundary = self._get_lane_group_boundary(token, "left", nusc_map)
            right_boundary = self._get_lane_group_boundary(token, "right", nusc_map)

            if baseline_path and left_boundary:
                left_boundary = align_boundary_direction(baseline_path, left_boundary)
            if baseline_path and right_boundary:
                right_boundary = align_boundary_direction(baseline_path, right_boundary)

            lane_group_data.append({
                "id": token,
                "lane_ids": lane_ids,
                "is_intersection": segment.get("is_intersection", False),
                "predecessor_lane_group_ids": incoming,
                "successor_lane_group_ids": outgoing,
                "left_boundary": left_boundary,
                "right_boundary": right_boundary,
                "geometry": row.geometry
            })

        return gpd.GeoDataFrame(lane_group_data, geometry="geometry")

    def _get_connected_segments(self, segment_token: str, nusc_map):
        incoming, outgoing = [], []

        if isinstance(nusc_map.lane_connector, dict):
            connectors = nusc_map.lane_connector.values()
        else: 
            connectors = nusc_map.lane_connector

        for connector in connectors:
            if connector.get("outgoing_lane") == segment_token:
                incoming.append(connector.get("incoming_lane"))
            elif connector.get("incoming_lane") == segment_token:
                outgoing.append(connector.get("outgoing_lane"))

        incoming = [id for id in incoming if id is not None]
        outgoing = [id for id in outgoing if id is not None]

        return incoming, outgoing

    def _get_lane_group_boundary(self, segment_token: str, side: str, nusc_map: NuScenesMap) -> Optional[LineString]:
        """Extract lane group boundary geometry."""
        boundary_type = "road_divider" if side == "left" else "lane_divider"
        if boundary_type not in self._gdf:
            return None

        # Find boundaries near the segment
        segment_geom = next(row.geometry for row in self._gdf["road_segment"].itertuples() if row.fid == segment_token)
        boundaries = self._gdf[boundary_type]

        # Find nearest boundary within 10 meters
        nearest = None
        min_dist = float('inf')
        for _, boundary_row in boundaries.iterrows():
            dist = segment_geom.distance(boundary_row.geometry)
            if dist < 10.0 and dist < min_dist:
                min_dist = dist
                nearest = boundary_row.geometry
        return nearest

    def _extract_intersection_dataframe(self) -> gpd.GeoDataFrame:
        """Create intersection GeoDataFrame with lane group information."""
        if "intersection" not in self._gdf:
            return gpd.GeoDataFrame()

        intersection_gdf = self._gdf["intersection"]
        intersections = []

        for _, row in intersection_gdf.iterrows():
            intersections.append({
                "id": row["fid"],
                "lane_group_ids": [],  # Not directly available
                "geometry": row.geometry
            })

        return gpd.GeoDataFrame(intersections)

    def _extract_crosswalk_dataframe(self) -> gpd.GeoDataFrame:
        """Create crosswalk GeoDataFrame."""
        if "ped_crossing" not in self._gdf:
            return gpd.GeoDataFrame()

        return self._gdf["ped_crossing"].rename(columns={"fid": "id"})[["id", "geometry"]]

    def _extract_walkway_dataframe(self) -> gpd.GeoDataFrame:
        """Create walkway GeoDataFrame."""
        if "walkway" not in self._gdf:
            return gpd.GeoDataFrame()

        return self._gdf["walkway"].rename(columns={"fid": "id"})[["id", "geometry"]]

    def _extract_carpark_dataframe(self) -> gpd.GeoDataFrame:
        """Create carpark GeoDataFrame."""
        if "carpark" not in self._gdf:
            return gpd.GeoDataFrame()

        return self._gdf["carpark"].rename(columns={"fid": "id"})[["id", "geometry"]]

    def _extract_generic_drivable_dataframe(self) -> gpd.GeoDataFrame:
        """Create generic drivable areas with unique IDs."""
        drivable_geoms = []
        drivable_ids = []

        # Combine road segments and lanes with layer prefixes
        for layer in ["road_segment", "lane"]:
            if layer in self._gdf:
                drivable_geoms.extend(self._gdf[layer].geometry.tolist())
                drivable_ids.extend([f"{layer}_{fid}" for fid in self._gdf[layer]["fid"].tolist()])

        # Add roads if available
        if "road" in self._gdf:
            drivable_geoms.extend(self._gdf["road"].geometry.tolist())
            drivable_ids.extend([f"road_{fid}" for fid in self._gdf["road"]["fid"].tolist()])

        if not drivable_geoms:
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame({"id": drivable_ids}, geometry=drivable_geoms)

    def _extract_stop_line_dataframe(self) -> gpd.GeoDataFrame:
        """Create stop line GeoDataFrame."""
        if "stop_line" not in self._gdf:
            return gpd.GeoDataFrame()
        return self._gdf["stop_line"].rename(columns={"fid": "id"})[["id", "geometry"]]

    def _extract_road_line_dataframe(self) -> gpd.GeoDataFrame:
        """Create road line GeoDataFrame with line types (road_divider + lane_divider)."""
        road_line_gdf = self._gdf.get("road_line", gpd.GeoDataFrame())
        if road_line_gdf.empty:
            return gpd.GeoDataFrame()

        if "line_type" not in road_line_gdf.columns:
            road_line_gdf["line_type"] = RoadLineType.UNKNOWN

        return road_line_gdf.rename(columns={"fid": "id"})[["id", "line_type", "geometry"]]
