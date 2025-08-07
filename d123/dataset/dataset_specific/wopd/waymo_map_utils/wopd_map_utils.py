from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom
from waymo_open_dataset import dataset_pb2

from d123.common.geometry.base import Point3DIndex
from d123.common.geometry.line.polylines import Polyline3D
from d123.common.geometry.units import mph_to_mps
from d123.dataset.dataset_specific.wopd.waymo_map_utils.womp_boundary_utils import extract_lane_boundaries
from d123.dataset.maps.map_datatypes import MapSurfaceType

# TODO:
# - Implement stop signs
# - Implement speed bumps
# - Implement driveways with a different semantic type if needed
# - Implement intersections and lane group logic
# - Handle lane type, e.g. TYPE_UNDEFINED = 0; TYPE_FREEWAY = 1; TYPE_SURFACE_STREET = 2; TYPE_BIKE_LANE = 3;


def convert_wopd_map(frame: dataset_pb2.Frame, map_file_path: Path) -> None:

    def _extract_polyline(data) -> npt.NDArray[np.float64]:
        polyline = np.array([[p.x, p.y, p.z] for p in data.polyline], dtype=np.float64)
        return polyline

    def _extract_polygon(data) -> npt.NDArray[np.float64]:
        polygon = np.array([[p.x, p.y, p.z] for p in data.polygon], dtype=np.float64)
        assert polygon.shape[0] >= 3, "Polygon must have at least 3 points"
        assert polygon.shape[1] == 3, "Polygon must have 3 coordinates (x, y, z)"
        return polygon

    lanes: Dict[int, npt.NDArray[np.float64]] = {}
    lanes_successors = defaultdict(list)
    lanes_predecessors = defaultdict(list)
    lanes_speed_limit_mps: Dict[int, float] = {}
    lanes_type: Dict[int, int] = {}

    road_lines: Dict[int, npt.NDArray[np.float64]] = {}
    road_edges: Dict[int, npt.NDArray[np.float64]] = {}
    crosswalks: Dict[int, npt.NDArray[np.float64]] = {}
    carparks: Dict[int, npt.NDArray[np.float64]] = {}

    for map_feature in frame.map_features:
        if map_feature.HasField("lane"):
            polyline = _extract_polyline(map_feature.lane)
            # Ignore lanes with less than 2 points or not 2D
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            lanes[map_feature.id] = polyline
            for lane_id_ in map_feature.lane.exit_lanes:
                lanes_successors[map_feature.id].append(lane_id_)
            for lane_id_ in map_feature.lane.exit_lanes:
                lanes_predecessors[map_feature.id].append(lane_id_)
            lanes_speed_limit_mps[map_feature.id] = mph_to_mps(map_feature.lane.speed_limit_mph)
            lanes_type[map_feature.id] = map_feature.lane.type
        elif map_feature.HasField("road_line"):
            polyline = _extract_polyline(map_feature.road_line)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            road_lines[map_feature.id] = polyline
        elif map_feature.HasField("road_edge"):
            polyline = _extract_polyline(map_feature.road_edge)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            road_edges[map_feature.id] = polyline
        elif map_feature.HasField("stop_sign"):
            # TODO: implement stop signs
            pass
        elif map_feature.HasField("crosswalk"):
            crosswalks[map_feature.id] = _extract_polygon(map_feature.crosswalk)
        elif map_feature.HasField("speed_bump"):
            # TODO: implement speed bumps
            pass
        elif map_feature.HasField("driveway"):
            # NOTE: Determine whether to use a different semantic type for driveways.
            carparks[map_feature.id] = _extract_polygon(map_feature.driveway)

    lane_left_boundaries_3d, lane_right_boundaries_3d = extract_lane_boundaries(
        lanes, lanes_successors, lanes_predecessors, road_lines, road_edges
    )

    lane_df = get_lane_df(
        lanes,
        lanes_successors,
        lanes_predecessors,
        lanes_speed_limit_mps,
        lane_left_boundaries_3d,
        lane_right_boundaries_3d,
    )
    lane_group_df = get_lane_group_df(
        lanes,
        lanes_successors,
        lanes_predecessors,
        lane_left_boundaries_3d,
        lane_right_boundaries_3d,
    )
    intersection_df = get_intersections_df()
    crosswalk_df = get_crosswalk_df(crosswalks)
    walkway_df = get_walkway_df()
    carpark_df = get_carpark_df(carparks)
    generic_drivable_df = get_generic_drivable_df()

    map_file_path.unlink(missing_ok=True)
    if not map_file_path.parent.exists():
        map_file_path.parent.mkdir(parents=True, exist_ok=True)

    lane_df.to_file(map_file_path, layer=MapSurfaceType.LANE.serialize(), driver="GPKG")
    lane_group_df.to_file(map_file_path, layer=MapSurfaceType.LANE_GROUP.serialize(), driver="GPKG", mode="a")
    intersection_df.to_file(map_file_path, layer=MapSurfaceType.INTERSECTION.serialize(), driver="GPKG", mode="a")
    crosswalk_df.to_file(map_file_path, layer=MapSurfaceType.CROSSWALK.serialize(), driver="GPKG", mode="a")
    walkway_df.to_file(map_file_path, layer=MapSurfaceType.WALKWAY.serialize(), driver="GPKG", mode="a")
    carpark_df.to_file(map_file_path, layer=MapSurfaceType.CARPARK.serialize(), driver="GPKG", mode="a")
    generic_drivable_df.to_file(
        map_file_path, layer=MapSurfaceType.GENERIC_DRIVABLE.serialize(), driver="GPKG", mode="a"
    )


def get_lane_df(
    lanes: Dict[int, npt.NDArray[np.float64]],
    lanes_successors: Dict[int, List[int]],
    lanes_predecessors: Dict[int, List[int]],
    lanes_speed_limit_mps: Dict[int, float],
    lanes_left_boundaries_3d: Dict[int, Polyline3D],
    lanes_right_boundaries_3d: Dict[int, Polyline3D],
) -> gpd.GeoDataFrame:

    ids = []
    lane_group_ids = []
    speed_limits_mps = []
    predecessor_ids = []
    successor_ids = []
    left_boundaries = []
    right_boundaries = []
    baseline_paths = []
    geometries = []

    for lane_id, lane_centerline_array in lanes.items():
        if lane_id not in lanes_left_boundaries_3d or lane_id not in lanes_right_boundaries_3d:
            continue
        lane_centerline = Polyline3D.from_array(lane_centerline_array)
        lane_speed_limit_mps = lanes_speed_limit_mps[lane_id] if lanes_speed_limit_mps[lane_id] > 0.0 else None

        ids.append(lane_id)
        lane_group_ids.append([lane_id])
        speed_limits_mps.append(lane_speed_limit_mps)
        predecessor_ids.append(lanes_predecessors[lane_id])
        successor_ids.append(lanes_successors[lane_id])
        left_boundaries.append(lanes_left_boundaries_3d[lane_id].linestring)
        right_boundaries.append(lanes_right_boundaries_3d[lane_id].linestring)
        baseline_paths.append(lane_centerline.linestring)

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
            "lane_group_id": lane_group_ids,
            "speed_limit_mps": speed_limits_mps,
            "predecessor_ids": predecessor_ids,
            "successor_ids": successor_ids,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
            "baseline_path": baseline_paths,
        }
    )

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


def get_generic_drivable_df() -> gpd.GeoDataFrame:
    ids = []
    geometries = []

    # NOTE: WOPD does not provide generic drivable areas, so we create an empty DataFrame.
    data = pd.DataFrame({"id": ids})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf
