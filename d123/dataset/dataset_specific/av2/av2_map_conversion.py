from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom
from flask import json

from d123.common.geometry.base import Point3DIndex
from d123.common.geometry.line.polylines import Polyline3D
from d123.dataset.maps.map_datatypes import MapLayer, RoadEdgeType, RoadLineType

# TODO:
# - TODO


def convert_av2_map(source_log_path: Path, map_file_path: Path) -> None:

    def _extract_polyline(data: List[Dict[str, float]]) -> Polyline3D:
        polyline = np.array([[p["x"], p["y"], p["z"]] for p in data], dtype=np.float64)
        return Polyline3D.from_array(polyline)

    map_folder = source_log_path / "map"

    next(map_folder.glob("*.npy"))
    log_map_archive_path = next(map_folder.glob("log_map_archive_*.json"))
    next(map_folder.glob("*img_Sim2_city.json*"))

    with open(log_map_archive_path, "r") as f:
        log_map_archive = json.load(f)

    defaultdict(list)
    drivable_areas: Dict[int, Polyline3D] = {}

    for drivable_area_id, drivable_area_dict in log_map_archive["drivable_areas"].items():
        # keys: ["area_boundary", "id"]
        drivable_areas[int(drivable_area_id)] = _extract_polyline(drivable_area_dict["area_boundary"])

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

    lane_df = get_lane_df(log_map_archive["lane_segments"])
    get_empty_gdf()
    get_empty_gdf()
    get_empty_gdf()
    get_empty_gdf()
    get_empty_gdf()
    generic_drivable_df = get_generic_drivable_df(drivable_areas)
    get_empty_gdf()
    get_empty_gdf()

    map_file_path.unlink(missing_ok=True)
    if not map_file_path.parent.exists():
        map_file_path.parent.mkdir(parents=True, exist_ok=True)

    lane_df.to_file(map_file_path, layer=MapLayer.LANE.serialize(), driver="GPKG")
    # lane_group_df.to_file(map_file_path, layer=MapLayer.LANE_GROUP.serialize(), driver="GPKG", mode="a")
    # intersection_df.to_file(map_file_path, layer=MapLayer.INTERSECTION.serialize(), driver="GPKG", mode="a")
    # crosswalk_df.to_file(map_file_path, layer=MapLayer.CROSSWALK.serialize(), driver="GPKG", mode="a")
    # walkway_df.to_file(map_file_path, layer=MapLayer.WALKWAY.serialize(), driver="GPKG", mode="a")
    # carpark_df.to_file(map_file_path, layer=MapLayer.CARPARK.serialize(), driver="GPKG", mode="a")
    generic_drivable_df.to_file(map_file_path, layer=MapLayer.GENERIC_DRIVABLE.serialize(), driver="GPKG", mode="a")
    # road_edge_df.to_file(map_file_path, layer=MapLayer.ROAD_EDGE.serialize(), driver="GPKG", mode="a")
    # road_line_df.to_file(map_file_path, layer=MapLayer.ROAD_LINE.serialize(), driver="GPKG", mode="a")


def get_lane_df(lanes: Dict[int, Any]) -> gpd.GeoDataFrame:

    ids = list(lanes.keys())
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
            left_boundary=lane_dict["left_lane_boundary"], right_boundary=lane_dict["right_lane_boundary"]
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


def get_empty_gdf() -> gpd.GeoDataFrame:
    ids = []
    geometries = []
    data = pd.DataFrame({"id": ids})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def get_lane_group_df(
    lanes: Dict[int, npt.NDArray[np.float64]],
    lanes_successors: Dict[int, List[int]],
    lanes_predecessors: Dict[int, List[int]],
    lanes_left_boundaries_3d: Dict[int, Polyline3D],
    lanes_right_boundaries_3d: Dict[int, Polyline3D],
) -> gpd.GeoDataFrame:

    ids = []
    lane_ids = []
    intersection_ids = []
    predecessor_lane_group_ids = []
    successor_lane_group_ids = []
    left_boundaries = []
    right_boundaries = []
    geometries = []

    # NOTE: WOPD does not provide lane groups, so we create a lane group for each lane.
    for lane_id in lanes.keys():
        if lane_id not in lanes_left_boundaries_3d or lane_id not in lanes_right_boundaries_3d:
            continue
        ids.append(lane_id)
        lane_ids.append([lane_id])
        intersection_ids.append(None)  # WOPD does not provide intersections
        predecessor_lane_group_ids.append(lanes_predecessors[lane_id])
        successor_lane_group_ids.append(lanes_successors[lane_id])
        left_boundaries.append(lanes_left_boundaries_3d[lane_id].linestring)
        right_boundaries.append(lanes_right_boundaries_3d[lane_id].linestring)
        geometry = geom.Polygon(
            np.vstack(
                [
                    lanes_left_boundaries_3d[lane_id].array[:, :2],
                    lanes_right_boundaries_3d[lane_id].array[:, :2][::-1],
                ]
            )
        )
        geometries.append(geometry)

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
    outlines = [geom.LineString(outline) for outline in crosswalks.values()]
    geometries = [geom.Polygon(outline[..., Point3DIndex.XY]) for outline in crosswalks.values()]

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


def get_road_edge_df(
    road_edges: Dict[int, npt.NDArray[np.float64]], road_edges_type: Dict[int, RoadEdgeType]
) -> gpd.GeoDataFrame:
    ids = list(road_edges.keys())
    geometries = [Polyline3D.from_array(road_edge).linestring for road_edge in road_edges.values()]

    data = pd.DataFrame(
        {
            "id": ids,
            "road_edge_type": [int(road_edge_type) for road_edge_type in road_edges_type.values()],
        }
    )
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


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
