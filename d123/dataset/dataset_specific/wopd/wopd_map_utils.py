from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom
from waymo_open_dataset import dataset_pb2

from d123.common.geometry.base import Point3DIndex
from d123.common.geometry.line.polylines import Polyline3D
from d123.dataset.maps.map_datatypes import MapSurfaceType


def convert_wopd_map(frame: dataset_pb2.Frame, map_file_path: Path) -> None:

    def _get_polyline(data) -> npt.NDArray[np.float64]:
        polyline = np.array([[p.x, p.y, p.z] for p in data.polyline], dtype=np.float64)
        return polyline

    def _get_polygon(data) -> npt.NDArray[np.float64]:
        polygon = np.array([[p.x, p.y, p.z] for p in data.polygon], dtype=np.float64)
        assert polygon.shape[0] >= 3, "Polygon must have at least 3 points"
        assert polygon.shape[1] == 3, "Polygon must have 3 coordinates (x, y, z)"
        return polygon

    lanes: Dict[int, npt.NDArray[np.float64]] = {}
    crosswalks: Dict[int, npt.NDArray[np.float64]] = {}
    carparks: Dict[int, npt.NDArray[np.float64]] = {}
    for map_feature in frame.map_features:
        if map_feature.HasField("lane"):
            polyline = _get_polyline(map_feature.lane)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            lanes[map_feature.id] = polyline
        elif map_feature.HasField("road_line"):
            pass
        elif map_feature.HasField("road_edge"):
            pass
        elif map_feature.HasField("stop_sign"):
            # TODO: implement stop signs
            pass
        elif map_feature.HasField("crosswalk"):
            crosswalks[map_feature.id] = _get_polygon(map_feature.crosswalk)
        elif map_feature.HasField("speed_bump"):
            # TODO: implement speed bumps
            pass
        elif map_feature.HasField("driveway"):
            # NOTE: Determine whether to use a different semantic type for driveways.
            carparks[map_feature.id] = _get_polygon(map_feature.driveway)

    lane_df = get_lane_df()
    lane_group_df = get_lane_group_df(lanes)
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


def get_lane_df() -> gpd.GeoDataFrame:

    ids = []
    lane_group_ids = []
    speed_limits_mps = []
    predecessor_ids = []
    successor_ids = []
    left_boundaries = []
    right_boundaries = []
    baseline_paths = []
    geometries = []

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


def get_lane_group_df(lanes: Dict[int, npt.NDArray[np.float64]]) -> gpd.GeoDataFrame:

    ids = []
    lane_ids = []
    intersection_ids = []
    predecessor_lane_group_ids = []
    successor_lane_group_ids = []
    left_boundaries = []
    right_boundaries = []
    geometries = []

    for lane_id, lane_array in lanes.items():
        if lane_array.ndim != 2 or lane_array.shape[0] < 2:
            continue

        centerline = Polyline3D.from_array(lane_array)
        left_boundary, right_boundary = create_lane_boundaries(centerline)
        lane_polygon = geom.Polygon(np.vstack([left_boundary.array[:, :2], right_boundary.array[:, :2][::-1]]))

        ids.append(lane_id)
        lane_ids.append([lane_id])
        intersection_ids.append([])
        predecessor_lane_group_ids.append([])
        successor_lane_group_ids.append([])
        left_boundaries.append(left_boundary.linestring)
        right_boundaries.append(right_boundary.linestring)
        geometries.append(lane_polygon)

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

    data = pd.DataFrame({"id": ids})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf


def create_lane_boundaries(polyline_3d: Polyline3D, width: float = 4) -> Tuple[Polyline3D, Polyline3D]:

    points = polyline_3d.array
    half_width = width / 2.0

    # Calculate the direction vectors between consecutive points
    directions = np.diff(points, axis=0)

    # Normalize the direction vectors
    directions_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions_normalized = directions / directions_norm

    # Calculate perpendicular vectors in the xy plane (z remains 0)
    perpendiculars = np.zeros_like(directions)
    perpendiculars[:, 0] = -directions_normalized[:, 1]  # -dy
    perpendiculars[:, 1] = directions_normalized[:, 0]  # dx

    # Create boundaries (need to handle the last point separately)
    left_boundary = points[:-1] + perpendiculars * half_width
    right_boundary = points[:-1] - perpendiculars * half_width

    # Handle the last point based on the last direction
    last_perp = perpendiculars[-1]
    left_boundary = np.vstack([left_boundary, points[-1] + last_perp * half_width])
    right_boundary = np.vstack([right_boundary, points[-1] - last_perp * half_width])

    return Polyline3D.from_array(left_boundary), Polyline3D.from_array(right_boundary)
