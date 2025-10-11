import logging
from collections import defaultdict
from typing import Dict, List, Set

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import LineString

from d123.conversion.utils.map_utils.road_edge.road_edge_2d_utils import get_road_edge_linear_rings
from d123.geometry import Point3DIndex
from d123.geometry.occupancy_map import OccupancyMap2D

logger = logging.getLogger(__name__)


def get_road_edges_3d_from_generic_drivable_area_df(generic_drivable_area_df: gpd.GeoDataFrame) -> List[LineString]:
    """
    Extracts 3D road edges from the generic drivable area GeoDataFrame.
    """
    # NOTE: this is a simplified version that assumes the generic drivable area covers all areas.
    # This is the case for argoverse 2.
    road_edges_2d = get_road_edge_linear_rings(generic_drivable_area_df.geometry.tolist())
    outlines = generic_drivable_area_df.outline.tolist()
    non_conflicting_road_edges = lift_road_edges_to_3d(road_edges_2d, outlines)
    return non_conflicting_road_edges


def get_road_edges_3d_from_gdf(
    lane_df: gpd.GeoDataFrame,
    carpark_df: gpd.GeoDataFrame,
    generic_drivable_area_df: gpd.GeoDataFrame,
    lane_group_df: gpd.GeoDataFrame,
) -> List[LineString]:

    # 1. Find conflicting lane groups, e.g. groups of lanes that overlap in 2D but have different Z-values (bridges)
    conflicting_lane_groups = _get_conflicting_lane_groups(lane_group_df, lane_df)

    # 2. Extract road edges in 2D (including conflicting lane groups)
    drivable_polygons = (
        lane_group_df.geometry.tolist() + carpark_df.geometry.tolist() + generic_drivable_area_df.geometry.tolist()
    )
    road_edges_2d = get_road_edge_linear_rings(drivable_polygons)

    # 3. Collect 3D boundaries of non-conflicting lane groups and other drivable areas
    non_conflicting_boundaries: List[LineString] = []
    for lane_group_id, lane_group_helper in lane_group_df.iterrows():
        if lane_group_id not in conflicting_lane_groups.keys():
            non_conflicting_boundaries.append(lane_group_helper["left_boundary"])
            non_conflicting_boundaries.append(lane_group_helper["right_boundary"])
    for outline in carpark_df.outline.tolist() + generic_drivable_area_df.outline.tolist():
        non_conflicting_boundaries.append(outline)

    # 4. Lift road edges to 3D using the boundaries of non-conflicting elements
    non_conflicting_road_edges = lift_road_edges_to_3d(road_edges_2d, non_conflicting_boundaries)

    # 5. Add road edges from conflicting lane groups
    resolved_road_edges = _resolve_conflicting_lane_groups(conflicting_lane_groups, lane_group_df)

    all_road_edges = non_conflicting_road_edges + resolved_road_edges

    return all_road_edges


def _get_conflicting_lane_groups(lane_group_df: gpd.GeoDataFrame, lane_df: gpd.GeoDataFrame) -> Dict[int, List[int]]:
    """
    Even more optimized version using vectorized operations where possible.
    """
    Z_THRESHOLD = 5.0  # [m] Z-value threshold for conflict detection

    # Convert to regular dictionaries for faster access
    lane_group_dict = lane_group_df.set_index("id").to_dict("index")
    lane_baseline_dict = dict(zip(lane_df.id.values, lane_df.baseline_path.values))

    # Pre-compute all centerlines
    centerlines_cache = {}
    polygons = []
    ids = []

    for lane_group_id, data in lane_group_dict.items():
        geometry = data["geometry"]
        lane_ids = data["lane_ids"]

        # Vectorized centerline computation
        centerlines = [np.array(lane_baseline_dict[lane_id].coords, dtype=np.float64) for lane_id in lane_ids]
        centerlines_3d = np.concatenate(centerlines, axis=0)

        centerlines_cache[lane_group_id] = centerlines_3d
        polygons.append(geometry)
        ids.append(lane_group_id)

    occupancy_map = OccupancyMap2D(polygons, ids)
    conflicting_lane_groups: Dict[int, List[int]] = defaultdict(list)
    processed_pairs = set()

    for i, lane_group_id in enumerate(ids):
        lane_group_polygon = polygons[i]
        lane_group_centerlines = centerlines_cache[lane_group_id]

        # Get all intersecting geometries at once
        intersecting_ids = occupancy_map.intersects(lane_group_polygon)
        intersecting_ids.remove(lane_group_id)

        for intersecting_id in intersecting_ids:
            pair_key = tuple(sorted([lane_group_id, intersecting_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            intersecting_geometry = occupancy_map[intersecting_id]
            if intersecting_geometry.geom_type != "Polygon":
                continue
            try:
                # Compute actual intersection for better centroid
                intersection = lane_group_polygon.intersection(intersecting_geometry)
            except shapely.errors.GEOSException as e:
                logger.debug(f"Error computing intersection for {pair_key}: {e}")
                continue

            if intersection.is_empty:
                continue

            intersection_centroid = np.array(intersection.centroid.coords[0], dtype=np.float64)
            intersecting_centerlines = centerlines_cache[intersecting_id]

            z_at_intersecting = _get_nearest_z_from_points_3d(intersecting_centerlines, intersection_centroid)
            z_at_lane_group = _get_nearest_z_from_points_3d(lane_group_centerlines, intersection_centroid)

            if np.abs(z_at_lane_group - z_at_intersecting) >= Z_THRESHOLD:
                conflicting_lane_groups[lane_group_id].append(intersecting_id)
                conflicting_lane_groups[intersecting_id].append(lane_group_id)

    return conflicting_lane_groups


def lift_road_edges_to_3d(
    road_edges_2d: List[shapely.LinearRing],
    boundaries: List[LineString],
    max_distance: float = 0.01,
) -> List[LineString]:
    """
    Even faster version using batch processing and optimized data structures.
    """
    if not road_edges_2d or not boundaries:
        return []

    # 1. Build comprehensive spatial index with all boundary segments
    boundary_segments = []

    for boundary_idx, boundary in enumerate(boundaries):
        coords = np.array(boundary.coords, dtype=np.float64).reshape(-1, 1, 3)
        segment_coords_boundary = np.concatenate([coords[:-1], coords[1:]], axis=1)
        boundary_segments.append(segment_coords_boundary)

    boundary_segments = np.concatenate(boundary_segments, axis=0)
    boundary_segment_linestrings = shapely.creation.linestrings(boundary_segments)

    occupancy_map = OccupancyMap2D(boundary_segment_linestrings)

    road_edges_3d = []
    for linear_ring in road_edges_2d:
        points_2d = np.array(linear_ring.coords, dtype=np.float64)
        points_3d = np.zeros((len(points_2d), 3), dtype=np.float64)
        points_3d[:, :2] = points_2d

        # 3. Batch query for all points
        query_points = shapely.creation.points(points_2d)
        results = occupancy_map.query_nearest(query_points, max_distance=max_distance, exclusive=True)

        for query_idx, geometry_idx in zip(*results):
            query_point = query_points[query_idx]
            segment_coords = boundary_segments[geometry_idx]
            best_z = _interpolate_z_on_segment(query_point, segment_coords)
            points_3d[query_idx, 2] = best_z

        continuous_segments = _find_continuous_segments(np.array(results[0]))

        for segment_indices in continuous_segments:
            if len(segment_indices) >= 2:
                segment_points = points_3d[segment_indices]
                road_edges_3d.append(LineString(segment_points))

    return road_edges_3d


def _get_nearest_z_from_points_3d(points_3d: npt.NDArray[np.float64], query_point: npt.NDArray[np.float64]) -> float:
    assert points_3d.ndim == 2 and points_3d.shape[1] == len(
        Point3DIndex
    ), "points_3d must be a 2D array with shape (N, 3)"
    distances = np.linalg.norm(points_3d[..., Point3DIndex.XY] - query_point[..., Point3DIndex.XY], axis=1)
    closest_point = points_3d[np.argmin(distances)]
    return closest_point[2]


def _interpolate_z_on_segment(point: shapely.Point, segment_coords: npt.NDArray[np.float64]) -> float:
    """Interpolate Z coordinate along a 3D line segment."""
    p1, p2 = segment_coords[0], segment_coords[1]

    # Project point onto segment
    segment_vec = p2[:2] - p1[:2]
    point_vec = np.array([point.x, point.y]) - p1[:2]

    # Handle degenerate case
    segment_length_sq = np.dot(segment_vec, segment_vec)
    if segment_length_sq == 0:
        return p1[2]

    # Calculate projection parameter
    t = np.dot(point_vec, segment_vec) / segment_length_sq
    t = np.clip(t, 0, 1)  # Clamp to segment bounds

    # Interpolate Z
    return p1[2] + t * (p2[2] - p1[2])


def _find_continuous_segments(indices: np.ndarray) -> List[np.ndarray]:
    """Vectorized version of finding continuous segments."""
    if len(indices) == 0:
        return []

    # Find breaks in continuity
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    segments = np.split(indices, breaks)

    # Filter segments with at least 2 points
    return [seg for seg in segments if len(seg) >= 2]


def _resolve_conflicting_lane_groups(
    conflicting_lane_groups: Dict[int, List[int]], lane_group_df: gpd.GeoDataFrame
) -> List[LineString]:

    # Split conflicting lane groups into non-conflicting sets for further merging
    non_conflicting_sets = _create_non_conflicting_sets(conflicting_lane_groups)

    road_edges_3d: List[LineString] = []
    for non_conflicting_set in non_conflicting_sets:

        # Collect 2D polygons of non-conflicting lane group set
        set_lane_group_rows = lane_group_df[lane_group_df.id.isin(non_conflicting_set)]
        connected_lane_group = []
        for row in set_lane_group_rows.itertuples():
            connected_lane_group.extend(row.predecessor_ids)
            connected_lane_group.extend(row.successor_ids)
        connected_lane_group_rows = lane_group_df[lane_group_df.id.isin(connected_lane_group)]

        set_polygons = set_lane_group_rows.geometry.tolist() + connected_lane_group_rows.geometry.tolist()

        # Get 2D road edge linestrings for the non-conflicting set
        set_road_edges_2d = get_road_edge_linear_rings(set_polygons)

        #  Collect 3D boundaries of non-conflicting lane groups
        set_boundaries_3d: List[LineString] = []
        for lane_group_id in non_conflicting_set:
            lane_group_helper = lane_group_df[lane_group_df.id == lane_group_id]
            set_boundaries_3d.append(lane_group_helper.left_boundary.values[0])
            set_boundaries_3d.append(lane_group_helper.right_boundary.values[0])

        # Lift road edges to 3D using the boundaries of non-conflicting lane groups
        lifted_road_edges_3d = lift_road_edges_to_3d(set_road_edges_2d, set_boundaries_3d)
        road_edges_3d.extend(lifted_road_edges_3d)

    return road_edges_3d


def _create_non_conflicting_sets(conflicts: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Creates sets of non-conflicting indices using NetworkX.
    """
    # Create graph from conflicts
    G = nx.Graph()
    for idx, conflict_list in conflicts.items():
        for conflict_idx in conflict_list:
            G.add_edge(idx, conflict_idx)

    result = []

    # Process each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)

        # Try bipartite coloring first (most common case)
        if nx.is_bipartite(subgraph):
            sets = nx.bipartite.sets(subgraph)
            result.extend([set(s) for s in sets])
        else:
            # Fall back to greedy coloring for non-bipartite graphs
            coloring = nx.greedy_color(subgraph, strategy="largest_first")
            color_groups = {}
            for node, color in coloring.items():
                if color not in color_groups:
                    color_groups[color] = set()
                color_groups[color].add(node)
            result.extend(color_groups.values())

    return result
