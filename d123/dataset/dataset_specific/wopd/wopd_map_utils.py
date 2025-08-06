from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom
from waymo_open_dataset import dataset_pb2

from d123.common.geometry.base import Point3D, Point3DIndex, StateSE2
from d123.common.geometry.line.polylines import Polyline3D
from d123.common.geometry.occupancy_map import OccupancyMap2D
from d123.common.geometry.transform.tranform_2d import translate_along_yaw
from d123.common.geometry.vector import Vector2D
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
    road_lines: Dict[int, npt.NDArray[np.float64]] = {}
    road_edges: Dict[int, npt.NDArray[np.float64]] = {}
    crosswalks: Dict[int, npt.NDArray[np.float64]] = {}
    carparks: Dict[int, npt.NDArray[np.float64]] = {}

    for map_feature in frame.map_features:
        if map_feature.HasField("lane"):
            polyline = _get_polyline(map_feature.lane)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            lanes[map_feature.id] = polyline
        elif map_feature.HasField("road_line"):
            polyline = _get_polyline(map_feature.road_line)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            road_lines[map_feature.id] = polyline
        elif map_feature.HasField("road_edge"):
            polyline = _get_polyline(map_feature.road_edge)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            road_edges[map_feature.id] = polyline
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

    lane_df = get_lane_df(lanes, road_lines, road_edges)
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


def get_lane_df(
    lanes: Dict[int, npt.NDArray[np.float64]],
    road_lines: Dict[int, npt.NDArray[np.float64]],
    road_edges: Dict[int, npt.NDArray[np.float64]],
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

    left_boundaries_3d, right_boundaries_3d = _extract_lane_boundaries(lanes, road_lines, road_edges)
    for lane_id, lane_centerline_array in lanes.items():
        if lane_id not in left_boundaries_3d or lane_id not in right_boundaries_3d:
            continue
        lane_centerline = Polyline3D.from_array(lane_centerline_array)

        ids.append(lane_id)
        lane_group_ids.append([lane_id])
        speed_limits_mps.append(None)  # TODO: Implement speed limits
        predecessor_ids.append([])  # TODO: Implement predecessor lanes
        successor_ids.append([])  # TODO: Implement successor lanes
        left_boundaries.append(left_boundaries_3d[lane_id].linestring)
        right_boundaries.append(right_boundaries_3d[lane_id].linestring)
        baseline_paths.append(lane_centerline.linestring)

        geometry = geom.Polygon(
            np.vstack([left_boundaries_3d[lane_id].array[:, :2], right_boundaries_3d[lane_id].array[:, :2][::-1]])
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


def _extract_lane_boundaries(
    lanes: Dict[int, npt.NDArray[np.float64]],
    road_lines: Dict[int, npt.NDArray[np.float64]],
    road_edges: Dict[int, npt.NDArray[np.float64]],
) -> Tuple[Dict[str, Polyline3D], Dict[str, Polyline3D]]:
    polyline_dict: Dict[str, Dict[int, Polyline3D]] = {"lane": {}, "roadline": {}, "roadedge": {}}

    for lane_id, lane_polyline in lanes.items():
        if lane_polyline.ndim == 2 and lane_polyline.shape[1] == 3 and len(lane_polyline) > 0:
            polyline_dict["lane"][lane_id] = Polyline3D.from_array(lane_polyline)

    # for road_line_id, road_line_polyline in road_lines.items():
    #     if road_line_polyline.ndim == 2 and road_line_polyline.shape[1] == 3 and len(road_line_polyline) > 0:
    #         polyline_dict["roadline"][road_line_id] = Polyline3D.from_array(road_line_polyline)

    for road_edge_id, road_edge_polyline in road_edges.items():
        if road_edge_polyline.ndim == 2 and road_edge_polyline.shape[1] == 3 and len(road_edge_polyline) > 0:
            polyline_dict["roadedge"][road_edge_id] = Polyline3D.from_array(road_edge_polyline)

    geometries = []
    tokens = []
    for line_type, polylines in polyline_dict.items():
        for polyline_id, polyline in polylines.items():
            geometries.append(polyline.polyline_2d.linestring)
            tokens.append(f"{line_type}_{polyline_id}")

    occupancy_2d = OccupancyMap2D(geometries, tokens)

    # for each lane
    #  - sample poses along centerline
    #  - query left and right perpendicular lines in 2d occupancy map
    #  - for {left, right} retrieve first perpendicular intersection with length MAX_HALF_WIDTH
    #   - if intersection with road edge, add intersecting point to {left, right} boundary
    #   - if intersection with centerline, add middle point to {left, right} boundary
    #   - if no intersection found, add MAX_HALF_WIDTH to {left, right} boundary

    MAX_LANE_WIDTH = 25.0  # meters
    MIN_LANE_WIDTH = 2.0
    DEFAULT_LANE_WIDTH = 4.0
    BOUNDARY_STEP_SIZE = 0.5  # meters
    MAX_Z_DISTANCE = 1.0  # meters

    left_boundaries = {lane_id: [] for lane_id in lanes.keys()}
    right_boundaries = {lane_id: [] for lane_id in lanes.keys()}

    def get_type_and_id_from_token(token: str) -> Tuple[str, int]:
        """Extract type and id from token."""
        line_type, line_id = token.split("_")
        return line_type, int(line_id)

    def get_polyline_from_token(token: str) -> Polyline3D:
        """Extract polyline from token."""
        line_type, line_id = get_type_and_id_from_token(token)
        return polyline_dict[line_type][line_id]

    def get_intersecting_point_3d(perp_linestring: geom.LineString, intersecting_token: str) -> Optional[Point3D]:

        intersecting_polyline_3d = get_polyline_from_token(intersecting_token)
        intersecting_linestring = occupancy_2d.geometries[occupancy_2d.token_to_idx[intersecting_token]]
        intersecting_point_2d = perp_linestring.intersection(intersecting_linestring)

        distance_2d_norm = None
        if isinstance(intersecting_point_2d, geom.Point):
            distance_2d_norm = intersecting_linestring.project(intersecting_point_2d, normalized=True)
        elif isinstance(intersecting_point_2d, geom.MultiPoint):
            geom_points = [point for point in intersecting_point_2d.geoms]
            starting_perp_point = geom.Point(perp_linestring.coords[0])
            start_perp_point_distance = np.array(
                [starting_perp_point.distance(geom_point) for geom_point in geom_points]
            )
            start_perp_point_distance[start_perp_point_distance < MIN_LANE_WIDTH / 2.0] = np.inf
            closest_index = np.argmin(start_perp_point_distance)
            distance_2d_norm = intersecting_linestring.project(geom_points[closest_index], normalized=True)

        intersecting_point_3d = None
        if distance_2d_norm is not None:
            intersecting_point_3d = intersecting_polyline_3d.interpolate(
                distance_2d_norm * intersecting_polyline_3d.length
            )
            if np.linalg.norm(intersecting_point_3d.point_2d.array - perp_linestring.coords[0]) < MIN_LANE_WIDTH / 2.0:
                return None

        return intersecting_point_3d

    for lane_id, lane_polyline in polyline_dict["lane"].items():
        current_lane_token = f"lane_{lane_id}"
        lane_polyline_se2 = lane_polyline.polyline_se2

        # 0. Find connected or intersecting lanes
        connected_lane_tokens = [
            token
            for token in occupancy_2d.intersects(lane_polyline_se2.linestring)
            if token.startswith("lane_")
            if token != current_lane_token
        ]

        # 1. sample poses along centerline
        distances_se2 = np.linspace(
            0, lane_polyline_se2.length, int(lane_polyline_se2.length / BOUNDARY_STEP_SIZE) + 1, endpoint=True
        )
        lane_query_se2 = [
            StateSE2.from_array(state_se2_array) for state_se2_array in lane_polyline_se2.interpolate(distances_se2)
        ]
        distances_3d = np.linspace(
            0, lane_polyline.length, int(lane_polyline.length / BOUNDARY_STEP_SIZE) + 1, endpoint=True
        )
        lane_query_3d = [
            Point3D.from_array(point_3d_array) for point_3d_array in lane_polyline.interpolate(distances_3d)
        ]
        assert len(lane_query_se2) == len(lane_query_3d)

        for lane_query_se2, lane_query_3d in zip(lane_query_se2, lane_query_3d):
            for sign in [1.0, -1.0]:
                perp_start_point = translate_along_yaw(lane_query_se2, Vector2D(0.0, sign * 0.1))
                perp_end_point = translate_along_yaw(lane_query_se2, Vector2D(0.0, sign * MAX_LANE_WIDTH / 2.0))
                perp_linestring = geom.LineString(
                    [[perp_start_point.x, perp_start_point.y], [perp_end_point.x, perp_end_point.y]]
                )

                # 1. find intersecting lines, compute 3D distance
                intersecting_tokens = occupancy_2d.intersects(perp_linestring)
                intersecting_points_3d: Dict[str, Point3D] = {}
                crosses_centerline = False

                for intersecting_token in intersecting_tokens:
                    # if intersecting_token in connected_lane_tokens:
                    #     crosses_centerline = True
                    #     continue
                    # if intersecting_token in connected_lane_tokens:
                    #     continue

                    # if intersecting_token == current_lane_token:
                    #     crosses_centerline = True

                    intersecting_point_3d = get_intersecting_point_3d(perp_linestring, intersecting_token)
                    if intersecting_point_3d is None:
                        continue

                    if np.abs(intersecting_point_3d.z - lane_query_3d.z) > MAX_Z_DISTANCE:
                        continue

                    intersecting_points_3d[intersecting_token] = intersecting_point_3d

                boundary_point_3d: Optional[Point3D] = None
                if len(intersecting_points_3d) > 0 and not crosses_centerline:
                    # 2. find closest intersection
                    intersecting_distances = {
                        token: np.linalg.norm(point_3d.array - lane_query_3d.array)
                        for token, point_3d in intersecting_points_3d.items()
                    }
                    closest_token = min(intersecting_distances, key=intersecting_distances.get)
                    closest_type, closest_id = get_type_and_id_from_token(closest_token)

                    if closest_type == "lane":
                        boundary_point_3d = Point3D.from_array(
                            (intersecting_points_3d[closest_token].array + lane_query_3d.array) / 2.0
                        )
                    elif closest_type == "roadedge":
                        boundary_point_3d = intersecting_points_3d[closest_token]
                else:
                    perp_boundary_distance = DEFAULT_LANE_WIDTH / 2.0  # Default to half the lane width
                    boundary_point_se2 = translate_along_yaw(
                        lane_query_se2, Vector2D(0.0, sign * perp_boundary_distance)
                    )
                    boundary_point_3d = Point3D(boundary_point_se2.x, boundary_point_se2.y, lane_query_3d.z)
                if sign == 1.0:
                    left_boundaries[lane_id].append(boundary_point_3d.array)
                else:
                    right_boundaries[lane_id].append(boundary_point_3d.array)

    left_boundaries = {
        lane_id: Polyline3D.from_array(np.array(boundary_array, dtype=np.float64))
        for lane_id, boundary_array in left_boundaries.items()
        if len(boundary_array) > 1
    }
    right_boundaries = {
        lane_id: Polyline3D.from_array(np.array(boundary_array, dtype=np.float64))
        for lane_id, boundary_array in right_boundaries.items()
        if len(boundary_array) > 1
    }

    return left_boundaries, right_boundaries
