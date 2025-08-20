from pathlib import Path
from typing import Any, Dict, Final, List

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom
from flask import json

from d123.common.geometry.base import Point3DIndex
from d123.common.geometry.line.polylines import Polyline3D
from d123.dataset.conversion.map.road_edge.road_edge_2d_utils import split_line_geometry_by_max_length
from d123.dataset.conversion.map.road_edge.road_edge_3d_utils import (
    get_road_edges_3d_from_generic_drivable_area_df,
)
from d123.dataset.maps.map_datatypes import MapLayer, RoadEdgeType, RoadLineType

# TODO:
# - TODO
LANE_GROUP_MARK_TYPES = ["DASHED_WHITE", "DOUBLE_DASH_WHITE", "DASH_SOLID_WHITE", "SOLID_DASH_WHITE", "SOLID_WHITE"]
MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # TODO: Add to config


def convert_av2_map(source_log_path: Path, map_file_path: Path) -> None:

    def _extract_polyline(data: List[Dict[str, float]], close: bool = False) -> Polyline3D:
        polyline = np.array([[p["x"], p["y"], p["z"]] for p in data], dtype=np.float64)
        if close:
            polyline = np.vstack([polyline, polyline[0]])

        return Polyline3D.from_array(polyline)

    map_folder = source_log_path / "map"

    next(map_folder.glob("*.npy"))
    log_map_archive_path = next(map_folder.glob("log_map_archive_*.json"))
    next(map_folder.glob("*img_Sim2_city.json*"))

    with open(log_map_archive_path, "r") as f:
        log_map_archive = json.load(f)

    drivable_areas: Dict[int, Polyline3D] = {}

    for drivable_area_id, drivable_area_dict in log_map_archive["drivable_areas"].items():
        # keys: ["area_boundary", "id"]
        drivable_areas[int(drivable_area_id)] = _extract_polyline(drivable_area_dict["area_boundary"], close=True)

    for lane_segment_id, lane_segment_dict in log_map_archive["lane_segments"].items():
        # keys = [
        #     "id",
        #     "is_intersection",
        #     "lane_type",
        #     "left_lane_boundary",
        #     "left_lane_mark_type",
        #     "right_lane_boundary",
        #     "right_lane_mark_type",
        #     "successors",
        #     "predecessors",
        #     "right_neighbor_id",
        #     "left_neighbor_id",
        # ]
        lane_segment_dict["left_lane_boundary"] = _extract_polyline(lane_segment_dict["left_lane_boundary"])
        lane_segment_dict["right_lane_boundary"] = _extract_polyline(lane_segment_dict["right_lane_boundary"])

    for crosswalk_id, crosswalk_dict in log_map_archive["pedestrian_crossings"].items():
        # keys = ["id", "outline"]
        # https://github.com/argoverse/av2-api/blob/6b22766247eda941cb1953d6a58e8d5631c561da/src/av2/map/pedestrian_crossing.py

        p1, p2 = np.array([[p["x"], p["y"], p["z"]] for p in crosswalk_dict["edge1"]], dtype=np.float64)
        p3, p4 = np.array([[p["x"], p["y"], p["z"]] for p in crosswalk_dict["edge2"]], dtype=np.float64)
        crosswalk_dict["outline"] = Polyline3D.from_array(np.array([p1, p2, p4, p3, p1], dtype=np.float64))

    lane_df = get_lane_df(log_map_archive["lane_segments"])
    lane_group_df = get_lane_group_df(log_map_archive["lane_segments"])
    get_empty_gdf()
    crosswalk_df = get_crosswalk_df(log_map_archive["pedestrian_crossings"])
    get_empty_gdf()
    get_empty_gdf()
    generic_drivable_df = get_generic_drivable_df(drivable_areas)
    road_edge_df = get_road_edge_df(generic_drivable_df)
    get_empty_gdf()

    map_file_path.unlink(missing_ok=True)
    if not map_file_path.parent.exists():
        map_file_path.parent.mkdir(parents=True, exist_ok=True)

    lane_df.to_file(map_file_path, layer=MapLayer.LANE.serialize(), driver="GPKG")
    lane_group_df.to_file(map_file_path, layer=MapLayer.LANE_GROUP.serialize(), driver="GPKG", mode="a")
    # intersection_df.to_file(map_file_path, layer=MapLayer.INTERSECTION.serialize(), driver="GPKG", mode="a")
    crosswalk_df.to_file(map_file_path, layer=MapLayer.CROSSWALK.serialize(), driver="GPKG", mode="a")
    # walkway_df.to_file(map_file_path, layer=MapLayer.WALKWAY.serialize(), driver="GPKG", mode="a")
    # carpark_df.to_file(map_file_path, layer=MapLayer.CARPARK.serialize(), driver="GPKG", mode="a")
    generic_drivable_df.to_file(map_file_path, layer=MapLayer.GENERIC_DRIVABLE.serialize(), driver="GPKG", mode="a")
    road_edge_df.to_file(map_file_path, layer=MapLayer.ROAD_EDGE.serialize(), driver="GPKG", mode="a")
    # road_line_df.to_file(map_file_path, layer=MapLayer.ROAD_LINE.serialize(), driver="GPKG", mode="a")


def get_empty_gdf() -> gpd.GeoDataFrame:
    ids = []
    outlines = []
    geometries = []
    data = pd.DataFrame({"id": ids, "outline": outlines})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_lane_df(lanes: Dict[int, Any]) -> gpd.GeoDataFrame:

    ids = [int(lane_id) for lane_id in lanes.keys()]
    lane_types = [0] * len(ids)  # TODO: Add lane types
    lane_group_ids = []
    speed_limits_mps = []
    predecessor_ids = []
    successor_ids = []
    left_boundaries = []
    right_boundaries = []
    left_lane_ids = []
    right_lane_ids = []
    baseline_paths = []
    geometries = []

    def _get_centerline_from_boundaries(
        left_boundary: Polyline3D, right_boundary: Polyline3D, resolution: float = 0.1
    ) -> Polyline3D:

        points_per_meter = 1 / resolution
        num_points = int(np.ceil(max([right_boundary.length, left_boundary.length]) * points_per_meter))
        right_array = right_boundary.interpolate(np.linspace(0, right_boundary.length, num_points, endpoint=True))
        left_array = left_boundary.interpolate(np.linspace(0, left_boundary.length, num_points, endpoint=True))

        return Polyline3D.from_array(np.mean([right_array, left_array], axis=0))

    for lane_id, lane_dict in lanes.items():
        # keys = [
        #     "id",
        #     "is_intersection",
        #     "lane_type",
        #     "left_lane_boundary",
        #     "left_lane_mark_type",
        #     "right_lane_boundary",
        #     "right_lane_mark_type",
        #     "successors",
        #     "predecessors",
        #     "right_neighbor_id",
        #     "left_neighbor_id",
        # ]
        lane_centerline = _get_centerline_from_boundaries(
            left_boundary=lane_dict["left_lane_boundary"],
            right_boundary=lane_dict["right_lane_boundary"],
        )
        lane_speed_limit_mps = None

        # ids.append(lane_id)
        # lane_types.append(lanes_type[lane_id])
        lane_group_ids.append(lane_id)
        speed_limits_mps.append(lane_speed_limit_mps)
        predecessor_ids.append(lane_dict["predecessors"])
        successor_ids.append(lane_dict["successors"])
        left_boundaries.append(lane_dict["left_lane_boundary"].linestring)
        right_boundaries.append(lane_dict["right_lane_boundary"].linestring)
        left_lane_ids.append(lane_dict["left_neighbor_id"])
        right_lane_ids.append(lane_dict["right_neighbor_id"])
        baseline_paths.append(lane_centerline.linestring)

        geometry = geom.Polygon(
            np.vstack(
                [
                    lane_dict["left_lane_boundary"].array[:, :2],
                    lane_dict["right_lane_boundary"].array[:, :2][::-1],
                ]
            )
        )
        geometries.append(geometry)

    data = pd.DataFrame(
        {
            "id": ids,
            "lane_type": lane_types,
            "lane_group_id": lane_group_ids,
            "speed_limit_mps": speed_limits_mps,
            "predecessor_ids": predecessor_ids,
            "successor_ids": successor_ids,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
            "left_lane_id": left_lane_ids,
            "right_lane_id": right_lane_ids,
            "baseline_path": baseline_paths,
        }
    )

    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_lane_group_df(lanes: Dict[int, Any]) -> gpd.GeoDataFrame:

    lane_group_sets = find_lane_groups(lanes)
    lane_group_set_dict = {i: lane_group for i, lane_group in enumerate(lane_group_sets)}

    ids = list(lane_group_set_dict.keys())
    lane_ids = []
    intersection_ids = []
    predecessor_lane_group_ids = []
    successor_lane_group_ids = []
    left_boundaries = []
    right_boundaries = []
    geometries = []

    def _get_lane_group_ids_of_lanes_ids(lane_ids: List[str]) -> List[int]:
        """Helper to find lane group ids that contain any of the given lane ids."""
        lane_group_ids_ = []
        for lane_group_id_, lane_group_set_ in lane_group_set_dict.items():
            if any(str(lane_id) in lane_group_set_ for lane_id in lane_ids):
                lane_group_ids_.append(lane_group_id_)
        return list(set(lane_group_ids_))

    for lane_group_id, lane_group_set in lane_group_set_dict.items():

        lane_ids.append([int(lane_id) for lane_id in lane_group_set])
        intersection_ids.append(None)  # NOTE: AV2 doesn't have explicit intersection objects.

        successor_lanes = []
        predecessor_lanes = []
        for lane_id in lane_group_set:
            lane_dict = lanes[str(lane_id)]
            successor_lanes.extend(lane_dict["successors"])
            predecessor_lanes.extend(lane_dict["predecessors"])

        left_boundary = lanes[lane_group_set[0]]["left_lane_boundary"]
        right_boundary = lanes[lane_group_set[-1]]["right_lane_boundary"]

        predecessor_lane_group_ids.append(_get_lane_group_ids_of_lanes_ids(predecessor_lanes))
        successor_lane_group_ids.append(_get_lane_group_ids_of_lanes_ids(successor_lanes))
        left_boundaries.append(left_boundary.linestring)
        right_boundaries.append(right_boundary.linestring)
        geometry = geom.Polygon(
            np.vstack(
                [
                    left_boundary.array[:, :2],
                    right_boundary.array[:, :2][::-1],
                ]
            )
        )
        geometries.append(geometry)

    data = pd.DataFrame(
        {
            "id": ids,
            "lane_ids": lane_ids,
            "intersection_id": intersection_ids,
            "predecessor_ids": predecessor_lane_group_ids,
            "successor_ids": successor_lane_group_ids,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
        }
    )
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_intersections_df() -> gpd.GeoDataFrame:
    ids = []
    lane_group_ids = []
    geometries = []

    # NOTE: WOPD does not provide intersections, so we create an empty DataFrame.
    data = pd.DataFrame({"id": ids, "lane_group_ids": lane_group_ids})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_carpark_df(carparks) -> gpd.GeoDataFrame:
    ids = list(carparks.keys())
    outlines = [geom.LineString(outline) for outline in carparks.values()]
    geometries = [geom.Polygon(outline[..., Point3DIndex.XY]) for outline in carparks.values()]

    data = pd.DataFrame({"id": ids, "outline": outlines})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_walkway_df() -> gpd.GeoDataFrame:
    ids = []
    geometries = []

    # NOTE: WOPD does not provide walkways, so we create an empty DataFrame.
    data = pd.DataFrame({"id": ids})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_crosswalk_df(crosswalks: Dict[int, npt.NDArray[np.float64]]) -> gpd.GeoDataFrame:
    ids = list(crosswalks.keys())
    outlines = []
    geometries = []
    for crosswalk_dict in crosswalks.values():
        outline = crosswalk_dict["outline"]
        outlines.append(outline.linestring)
        geometries.append(geom.Polygon(outline.array[:, Point3DIndex.XY]))

    data = pd.DataFrame({"id": ids, "outline": outlines})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_generic_drivable_df(drivable_areas: Dict[int, Polyline3D]) -> gpd.GeoDataFrame:
    ids = list(drivable_areas.keys())
    outlines = [drivable_area.linestring for drivable_area in drivable_areas.values()]
    geometries = [geom.Polygon(drivable_area.array[:, Point3DIndex.XY]) for drivable_area in drivable_areas.values()]

    data = pd.DataFrame({"id": ids, "outline": outlines})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_road_edge_df(generic_drivable_area_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    road_edges = get_road_edges_3d_from_generic_drivable_area_df(generic_drivable_area_df)
    road_edges = split_line_geometry_by_max_length(road_edges, MAX_ROAD_EDGE_LENGTH)

    ids = np.arange(len(road_edges), dtype=np.int64).tolist()
    # TODO @DanielDauner: Figure out if other types should/could be assigned here.
    road_edge_types = [int(RoadEdgeType.ROAD_EDGE_BOUNDARY)] * len(road_edges)
    geometries = road_edges
    return gpd.GeoDataFrame(pd.DataFrame({"id": ids, "road_edge_type": road_edge_types}), geometry=geometries)


def get_road_line_df(
    road_lines: Dict[int, npt.NDArray[np.float64]], road_lines_type: Dict[int, RoadLineType]
) -> gpd.GeoDataFrame:
    ids = list(road_lines.keys())
    geometries = [Polyline3D.from_array(road_edge).linestring for road_edge in road_lines.values()]

    data = pd.DataFrame(
        {
            "id": ids,
            "road_line_type": [int(road_line_type) for road_line_type in road_lines_type.values()],
        }
    )
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def find_lane_groups(lanes) -> List[List[str]]:

    visited = set()
    lane_groups = []

    def _get_valid_neighbor_id(lane_data, direction):
        """Helper function to safely get neighbor ID"""
        neighbor_key = f"{direction}_neighbor_id"
        neighbor_id = str(lane_data.get(neighbor_key))
        mark_type = lane_data.get(f"{direction}_lane_mark_type", None)

        if (neighbor_id is not None) and (neighbor_id in lanes) and (mark_type in LANE_GROUP_MARK_TYPES):
            return neighbor_id
        return None

    def _traverse_group(start_lane_id):
        """
        Traverse left and right from a starting lane to find all connected parallel lanes
        """
        group = [start_lane_id]
        queue = [start_lane_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue

            visited.add(current_id)

            # Check left neighbor
            left_neighbor = _get_valid_neighbor_id(lanes[current_id], "left")
            if left_neighbor is not None and left_neighbor not in visited:
                queue.append(left_neighbor)
                group = [left_neighbor] + group

            # Check right neighbor
            right_neighbor = _get_valid_neighbor_id(lanes[current_id], "right")
            if right_neighbor is not None and right_neighbor not in visited:
                queue.append(right_neighbor)
                group = group + [right_neighbor]

        return group

    # Find all lane groups
    for lane_id in lanes:
        if lane_id not in visited:
            group = _traverse_group(lane_id)
            lane_groups.append(group)

    return lane_groups
