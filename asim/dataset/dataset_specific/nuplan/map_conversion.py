import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import LineString

from asim.dataset.maps.gpkg.utils import get_all_rows_with_value, get_row_with_value
from asim.dataset.maps.map_datatypes import MapSurfaceType

MAP_FILES = {
    "sg-one-north": "sg-one-north/9.17.1964/map.gpkg",
    "us-ma-boston": "us-ma-boston/9.12.1817/map.gpkg",
    "us-nv-las-vegas-strip": "us-nv-las-vegas-strip/9.15.1915/map.gpkg",
    "us-pa-pittsburgh-hazelwood": "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg",
}

NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
MAP_LOCATIONS = {"sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood"}
GPKG_LAYERS: List[str] = [
    "baseline_paths",
    "carpark_areas",
    "generic_drivable_areas",
    "dubins_nodes",
    "lane_connectors",
    "intersections",
    "boundaries",
    "crosswalks",
    "lanes_polygons",
    "lane_group_connectors",
    "lane_groups_polygons",
    "road_segments",
    "stop_polygons",
    "traffic_lights",
    "walkways",
    "gen_lane_connectors_scaled_width_polygons",
]


class NuPlanMapConverter:
    def __init__(self, map_path: Path) -> None:

        self._map_path: Path = map_path
        self._gdf: Optional[Dict[str, gpd.GeoDataFrame]] = None
        self._extract_lane_group_boundaries_from_lanes: bool = False

    def convert(self, map_name: str = "us-pa-pittsburgh-hazelwood") -> None:
        assert map_name in MAP_LOCATIONS, f"Map name {map_name} is not supported."

        map_file_path = Path(NUPLAN_MAPS_ROOT) / MAP_FILES[map_name]
        self._load_dataframes(map_file_path)

        lane_df = self._extract_lane_dataframe()
        lane_group_df = self._extract_lane_group_dataframe()
        intersection_df = self._extract_intersection_dataframe()
        crosswalk_df = self._extract_crosswalk_dataframe()
        walkway_df = self._extract_walkway_dataframe()
        carpark_df = self._extract_carpark_dataframe()
        generic_drivable_df = self._extract_generic_drivable_dataframe()

        if not self._map_path.exists():
            self._map_path.mkdir(parents=True, exist_ok=True)

        map_file_name = self._map_path / f"nuplan_{map_name}.gpkg"
        lane_df.to_file(map_file_name, layer=MapSurfaceType.LANE.serialize(), driver="GPKG")
        lane_group_df.to_file(map_file_name, layer=MapSurfaceType.LANE_GROUP.serialize(), driver="GPKG", mode="a")
        intersection_df.to_file(map_file_name, layer=MapSurfaceType.INTERSECTION.serialize(), driver="GPKG", mode="a")
        crosswalk_df.to_file(map_file_name, layer=MapSurfaceType.CROSSWALK.serialize(), driver="GPKG", mode="a")
        walkway_df.to_file(map_file_name, layer=MapSurfaceType.WALKWAY.serialize(), driver="GPKG", mode="a")
        carpark_df.to_file(map_file_name, layer=MapSurfaceType.CARPARK.serialize(), driver="GPKG", mode="a")
        generic_drivable_df.to_file(
            map_file_name, layer=MapSurfaceType.GENERIC_DRIVABLE.serialize(), driver="GPKG", mode="a"
        )

    def _load_dataframes(self, map_file_path: Path) -> None:

        # The projected coordinate system depends on which UTM zone the mapped location is in.
        map_meta = gpd.read_file(map_file_path, layer="meta", engine="pyogrio")
        projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

        self._gdf = {}
        for layer_name in GPKG_LAYERS:
            with warnings.catch_warnings():
                # Suppress the warnings from the GPKG operations below so that they don't spam the training logs.
                warnings.filterwarnings("ignore")

                gdf_in_pixel_coords = pyogrio.read_dataframe(map_file_path, layer=layer_name, fid_as_index=True)
                gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)
                # gdf_in_utm_coords = gdf_in_pixel_coords

                # For backwards compatibility, cast the index to string datatype.
                #   and mirror it to the "fid" column.
                gdf_in_utm_coords.index = gdf_in_utm_coords.index.map(str)
                gdf_in_utm_coords["fid"] = gdf_in_utm_coords.index

            self._gdf[layer_name] = gdf_in_utm_coords

    def _extract_lane_dataframe(self) -> gpd.GeoDataFrame:
        assert self._gdf is not None, "Call `.initialize()` before retrieving data!"
        lane_df = self._extract_nuplan_lane_dataframe()
        lane_connector_df = self._extract_nuplan_lane_connector_dataframe()
        combined_df = pd.concat([lane_df, lane_connector_df], ignore_index=True)
        return combined_df

    def _extract_nuplan_lane_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: lane_index (?), creator_id, name (?), road_type_fid (?), lane_type_fid (?), width (?), left_offset (?), right_offset (?),
        # min_speed (?), max_speed (?), stops, left_has_reflectors (?), right_has_reflectors (?), from_edge_fid, to_edge_fid

        ids = self._gdf["lanes_polygons"].lane_fid.to_list()
        lane_group_ids = self._gdf["lanes_polygons"].lane_group_fid.to_list()
        speed_limits_mps = self._gdf["lanes_polygons"].speed_limit_mps.to_list()
        predecessor_ids = []
        successor_ids = []
        left_boundaries = []
        right_boundaries = []
        baseline_paths = []
        geometries = self._gdf["lanes_polygons"].geometry.to_list()

        for lane_id in ids:

            # 1. predecessor_ids, successor_ids
            _predecessor_ids = get_all_rows_with_value(
                self._gdf["lane_connectors"],
                "entry_lane_fid",
                lane_id,
            )["fid"].tolist()
            _successor_ids = get_all_rows_with_value(
                self._gdf["lane_connectors"],
                "exit_lane_fid",
                lane_id,
            )["fid"].tolist()
            predecessor_ids.append(_predecessor_ids)
            successor_ids.append(_successor_ids)

            # 2. left_boundaries, right_boundaries
            lane_series = get_row_with_value(self._gdf["lanes_polygons"], "fid", str(lane_id))
            left_boundary_fid = lane_series["left_boundary_fid"]
            left_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]

            right_boundary_fid = lane_series["right_boundary_fid"]
            right_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]

            # 3. baseline_paths
            baseline_path = get_row_with_value(self._gdf["baseline_paths"], "lane_fid", float(lane_id))["geometry"]

            left_boundary = align_boundary_direction(baseline_path, left_boundary)
            right_boundary = align_boundary_direction(baseline_path, right_boundary)

            left_boundaries.append(left_boundary)
            right_boundaries.append(right_boundary)
            baseline_paths.append(baseline_path)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_group_id": lane_group_ids,
                "speed_limits_mps": speed_limits_mps,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
                "baseline_path": baseline_paths,
            }
        )

        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_nuplan_lane_connector_dataframe(self) -> None:
        # NOTE: drops: exit_lane_group_fid, entry_lane_group_fid, to_edge_fid,
        # turn_type_fid (?), bulb_fids (?), traffic_light_stop_line_fids (?), overlap (?), creator_id
        # left_has_reflectors (?), right_has_reflectors (?)
        ids = self._gdf["lane_connectors"].fid.to_list()
        lane_group_ids = self._gdf["lane_connectors"].lane_group_connector_fid.to_list()
        speed_limits_mps = self._gdf["lane_connectors"].speed_limit_mps.to_list()
        predecessor_ids = []
        successor_ids = []
        left_boundaries = []
        right_boundaries = []
        baseline_paths = []
        geometries = []

        for lane_id in ids:
            # 1. predecessor_ids, successor_ids
            lane_connector_row = get_row_with_value(self._gdf["lane_connectors"], "fid", str(lane_id))
            predecessor_ids.append([lane_connector_row["entry_lane_fid"]])
            successor_ids.append([lane_connector_row["exit_lane_fid"]])

            # 2. left_boundaries, right_boundaries
            lane_connector_polygons_row = get_row_with_value(
                self._gdf["gen_lane_connectors_scaled_width_polygons"], "lane_connector_fid", str(lane_id)
            )
            left_boundary_fid = lane_connector_polygons_row["left_boundary_fid"]
            left_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]

            right_boundary_fid = lane_connector_polygons_row["right_boundary_fid"]
            right_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]

            # 3. baseline_paths
            baseline_path = get_row_with_value(self._gdf["baseline_paths"], "lane_connector_fid", float(lane_id))[
                "geometry"
            ]

            left_boundary = align_boundary_direction(baseline_path, left_boundary)
            right_boundary = align_boundary_direction(baseline_path, right_boundary)

            left_boundaries.append(left_boundary)
            right_boundaries.append(right_boundary)
            baseline_paths.append(baseline_path)

            # 4. geometries
            geometries.append(lane_connector_polygons_row.geometry)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_group_id": lane_group_ids,
                "speed_limits_mps": speed_limits_mps,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
                "baseline_path": baseline_paths,
            }
        )

        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_lane_group_dataframe(self) -> gpd.GeoDataFrame:
        lane_group_df = self._extract_nuplan_lane_group_dataframe()
        lane_connector_group_df = self._extract_nuplan_lane_connector_group_dataframe()
        combined_df = pd.concat([lane_group_df, lane_connector_group_df], ignore_index=True)
        return combined_df

    def _extract_nuplan_lane_group_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: creator_id, from_edge_fid, to_edge_fid
        ids = self._gdf["lane_groups_polygons"].fid.to_list()
        lane_ids = []
        intersection_ids = [None] * len(ids)
        predecessor_lane_group_ids = []
        successor_lane_group_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = self._gdf["lane_groups_polygons"].geometry.to_list()

        for lane_group_id in ids:
            # 1. lane_ids
            lane_ids_ = get_all_rows_with_value(
                self._gdf["lanes_polygons"],
                "lane_group_fid",
                lane_group_id,
            )["fid"].tolist()
            lane_ids.append(lane_ids_)

            # 2. predecessor_lane_group_ids, successor_lane_group_ids
            predecessor_lane_group_ids_ = get_all_rows_with_value(
                self._gdf["lane_group_connectors"],
                "to_lane_group_fid",
                lane_group_id,
            )["fid"].tolist()
            successor_lane_group_ids_ = get_all_rows_with_value(
                self._gdf["lane_group_connectors"],
                "from_lane_group_fid",
                lane_group_id,
            )["fid"].tolist()
            predecessor_lane_group_ids.append(predecessor_lane_group_ids_)
            successor_lane_group_ids.append(successor_lane_group_ids_)

            # 3. left_boundaries, right_boundaries
            if self._extract_lane_group_boundaries_from_lanes:
                lane_rows = [
                    get_row_with_value(self._gdf["lanes_polygons"], "fid", str(lane_id)) for lane_id in lane_ids_
                ]
                lane_rows = sorted(lane_rows, key=lambda x: int(x["lane_index"]))
                left_lane_row = lane_rows[0]
                right_lane_row = lane_rows[-1]

                left_boundary_fid = left_lane_row["left_boundary_fid"]
                left_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]

                right_boundary_fid = right_lane_row["right_boundary_fid"]
                right_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]

                # pass

            else:
                # pass
                lane_group_row = get_row_with_value(self._gdf["lane_groups_polygons"], "fid", str(lane_group_id))
                left_boundary_fid = lane_group_row["left_boundary_fid"]
                left_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]

                right_boundary_fid = lane_group_row["right_boundary_fid"]
                right_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]

            left_boundaries.append(left_boundary)
            right_boundaries.append(right_boundary)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_ids": lane_ids,
                "intersection_id": intersection_ids,
                "predecessor_lane_group_ids": predecessor_lane_group_ids,
                "successor_lane_group_ids": successor_lane_group_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_nuplan_lane_connector_group_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: creator_id, from_edge_fid, to_edge_fid, intersection_fid
        ids = self._gdf["lane_group_connectors"].fid.to_list()
        lane_ids = []
        intersection_ids = self._gdf["lane_group_connectors"].intersection_fid.to_list()
        predecessor_lane_group_ids = []
        successor_lane_group_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = self._gdf["lane_group_connectors"].geometry.to_list()

        for lane_group_connector_id in ids:
            # 1. lane_ids
            lane_ids_ = get_all_rows_with_value(
                self._gdf["lane_connectors"], "lane_group_connector_fid", lane_group_connector_id
            )["fid"].tolist()
            lane_ids.append(lane_ids_)

            # 2. predecessor_lane_group_ids, successor_lane_group_ids
            lane_group_connector_row = get_row_with_value(
                self._gdf["lane_group_connectors"], "fid", lane_group_connector_id
            )
            predecessor_lane_group_ids.append([str(lane_group_connector_row["from_lane_group_fid"])])
            successor_lane_group_ids.append([str(lane_group_connector_row["to_lane_group_fid"])])

            # 3. left_boundaries, right_boundaries
            left_boundary_fid = lane_group_connector_row["left_boundary_fid"]
            left_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]
            right_boundary_fid = lane_group_connector_row["right_boundary_fid"]
            right_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]
            left_boundaries.append(left_boundary)
            right_boundaries.append(right_boundary)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_ids": lane_ids,
                "intersection_id": intersection_ids,
                "predecessor_lane_group_ids": predecessor_lane_group_ids,
                "successor_lane_group_ids": successor_lane_group_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_intersection_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: creator_id, intersection_type_fid (?), is_mini (?)
        ids = self._gdf["intersections"].fid.to_list()
        lane_group_ids = []
        for intersection_id in ids:
            lane_group_connector_ids = get_all_rows_with_value(
                self._gdf["lane_group_connectors"], "intersection_fid", str(intersection_id)
            )["fid"].tolist()
            lane_group_ids.append(lane_group_connector_ids)
        data = pd.DataFrame({"id": ids, "lane_group_ids": lane_group_ids})
        return gpd.GeoDataFrame(data, geometry=self._gdf["intersections"].geometry.to_list())

    def _extract_crosswalk_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: creator_id, intersection_fids, lane_fids, is_marked (?)
        data = pd.DataFrame({"id": self._gdf["crosswalks"].fid.to_list()})
        return gpd.GeoDataFrame(data, geometry=self._gdf["crosswalks"].geometry.to_list())

    def _extract_walkway_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: creator_id
        data = pd.DataFrame({"id": self._gdf["walkways"].fid.to_list()})
        return gpd.GeoDataFrame(data, geometry=self._gdf["walkways"].geometry.to_list())

    def _extract_carpark_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: heading, creator_id
        data = pd.DataFrame({"id": self._gdf["carpark_areas"].fid.to_list()})
        return gpd.GeoDataFrame(data, geometry=self._gdf["carpark_areas"].geometry.to_list())

    def _extract_generic_drivable_dataframe(self) -> gpd.GeoDataFrame:
        # NOTE: drops: creator_id
        data = pd.DataFrame({"id": self._gdf["generic_drivable_areas"].fid.to_list()})
        return gpd.GeoDataFrame(data, geometry=self._gdf["generic_drivable_areas"].geometry.to_list())


def flip_linestring(linestring: LineString) -> LineString:
    return LineString(linestring.coords[::-1])


def lines_same_direction(centerline: LineString, boundary: LineString) -> bool:
    center_start = np.array(centerline.coords[0])
    center_end = np.array(centerline.coords[-1])
    boundary_start = np.array(boundary.coords[0])
    boundary_end = np.array(boundary.coords[-1])

    # Distance from centerline start to boundary start + centerline end to boundary end
    same_dir_dist = np.linalg.norm(center_start - boundary_start) + np.linalg.norm(center_end - boundary_end)
    opposite_dir_dist = np.linalg.norm(center_start - boundary_end) + np.linalg.norm(center_end - boundary_start)

    return same_dir_dist <= opposite_dir_dist


def align_boundary_direction(centerline: LineString, boundary: LineString) -> LineString:
    if not lines_same_direction(centerline, boundary):
        return flip_linestring(boundary)
    return boundary
