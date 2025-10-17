import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Final, List

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import polygonize, unary_union

from py123d.conversion.utils.map_utils.opendrive.parser.opendrive import Junction, OpenDrive
from py123d.conversion.utils.map_utils.opendrive.utils.collection import collect_element_helpers
from py123d.conversion.utils.map_utils.opendrive.utils.id_mapping import IntIDMapping
from py123d.conversion.utils.map_utils.opendrive.utils.lane_helper import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
)
from py123d.conversion.utils.map_utils.opendrive.utils.objects_helper import (
    OpenDriveObjectHelper,
)
from py123d.conversion.utils.map_utils.road_edge.road_edge_2d_utils import split_line_geometry_by_max_length
from py123d.conversion.utils.map_utils.road_edge.road_edge_3d_utils import get_road_edges_3d_from_gdf
from py123d.datatypes.maps.map_datatypes import MapLayer, RoadEdgeType, RoadLineType

logger = logging.getLogger(__name__)
PY123D_MAPS_ROOT = Path(os.environ.get("PY123D_MAPS_ROOT"))

MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]


def convert_from_xodr(
    xordr_file: Path,
    location: str,
    interpolation_step_size: float,
    connection_distance_threshold: float,
) -> None:

    opendrive = OpenDrive.parse_from_file(xordr_file)

    _, junction_dict, lane_helper_dict, lane_group_helper_dict, object_helper_dict = collect_element_helpers(
        opendrive, interpolation_step_size, connection_distance_threshold
    )

    # Collect data frames and store
    dataframes: Dict[MapLayer, gpd.GeoDataFrame] = {}
    dataframes[MapLayer.LANE] = _extract_lane_dataframe(lane_group_helper_dict)
    dataframes[MapLayer.LANE_GROUP] = _extract_lane_group_dataframe(lane_group_helper_dict)
    dataframes[MapLayer.WALKWAY] = _extract_walkways_dataframe(lane_helper_dict)
    dataframes[MapLayer.CARPARK] = _extract_carpark_dataframe(lane_helper_dict)
    dataframes[MapLayer.GENERIC_DRIVABLE] = _extract_generic_drivable_dataframe(lane_helper_dict)
    dataframes[MapLayer.INTERSECTION] = _extract_intersections_dataframe(junction_dict, lane_group_helper_dict)
    dataframes[MapLayer.CROSSWALK] = _extract_crosswalk_dataframe(object_helper_dict)

    _convert_ids_to_int(
        dataframes[MapLayer.LANE],
        dataframes[MapLayer.WALKWAY],
        dataframes[MapLayer.CARPARK],
        dataframes[MapLayer.GENERIC_DRIVABLE],
        dataframes[MapLayer.LANE_GROUP],
        dataframes[MapLayer.INTERSECTION],
        dataframes[MapLayer.CROSSWALK],
    )
    dataframes[MapLayer.ROAD_EDGE] = _extract_road_edge_df(
        dataframes[MapLayer.LANE],
        dataframes[MapLayer.CARPARK],
        dataframes[MapLayer.GENERIC_DRIVABLE],
        dataframes[MapLayer.LANE_GROUP],
    )
    dataframes[MapLayer.ROAD_LINE] = _extract_road_line_df(
        dataframes[MapLayer.LANE],
        dataframes[MapLayer.LANE_GROUP],
    )
    map_file_name = PY123D_MAPS_ROOT / f"{location}.gpkg"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'crs' was not provided")
        for layer, gdf in dataframes.items():
            gdf.to_file(map_file_name, layer=layer.serialize(), driver="GPKG", mode="a")


def _extract_lane_dataframe(lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper]) -> gpd.GeoDataFrame:

    ids = []
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

    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_id = lane_group_helper.lane_group_id
        lane_helpers = lane_group_helper.lane_helpers
        num_lanes = len(lane_helpers)
        # NOTE: Lanes are going left to right, ie. inner to outer
        for lane_idx, lane_helper in enumerate(lane_helpers):
            ids.append(lane_helper.lane_id)
            lane_group_ids.append(lane_group_id)
            speed_limits_mps.append(lane_helper.speed_limit_mps)
            predecessor_ids.append(lane_helper.predecessor_lane_ids)
            successor_ids.append(lane_helper.successor_lane_ids)
            left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
            right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
            baseline_paths.append(shapely.LineString(lane_helper.center_polyline_3d))
            geometries.append(lane_helper.shapely_polygon)
            left_lane_id = lane_helpers[lane_idx - 1].lane_id if lane_idx > 0 else None
            right_lane_id = lane_helpers[lane_idx + 1].lane_id if lane_idx < num_lanes - 1 else None
            left_lane_ids.append(left_lane_id)
            right_lane_ids.append(right_lane_id)

    data = pd.DataFrame(
        {
            "id": ids,
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
    return gpd.GeoDataFrame(data, geometry=geometries)


def _extract_lane_group_dataframe(lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper]) -> gpd.GeoDataFrame:

    ids = []
    lane_ids = []
    predecessor_lane_group_ids = []
    successor_lane_group_ids = []
    intersection_ids = []
    left_boundaries = []
    right_boundaries = []
    geometries = []

    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_helper: OpenDriveLaneGroupHelper
        ids.append(lane_group_helper.lane_group_id)
        lane_ids.append([lane_helper.lane_id for lane_helper in lane_group_helper.lane_helpers])
        predecessor_lane_group_ids.append(lane_group_helper.predecessor_lane_group_ids)
        successor_lane_group_ids.append(lane_group_helper.successor_lane_group_ids)
        intersection_ids.append(lane_group_helper.junction_id)
        left_boundaries.append(shapely.LineString(lane_group_helper.inner_polyline_3d))
        right_boundaries.append(shapely.LineString(lane_group_helper.outer_polyline_3d))
        geometries.append(lane_group_helper.shapely_polygon)

    data = pd.DataFrame(
        {
            "id": ids,
            "lane_ids": lane_ids,
            "predecessor_ids": predecessor_lane_group_ids,
            "successor_ids": successor_lane_group_ids,
            "intersection_id": intersection_ids,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
        }
    )
    gdf = gpd.GeoDataFrame(data, geometry=geometries)

    return gdf


def _extract_walkways_dataframe(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> gpd.GeoDataFrame:

    ids = []
    left_boundaries = []
    right_boundaries = []
    outlines = []
    geometries = []

    for lane_helper in lane_helper_dict.values():
        if lane_helper.type == "sidewalk":
            ids.append(lane_helper.lane_id)
            left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
            right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
            outlines.append(shapely.LineString(lane_helper.outline_polyline_3d))
            geometries.append(lane_helper.shapely_polygon)

    data = pd.DataFrame(
        {
            "id": ids,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
            "outline": outlines,
        }
    )
    return gpd.GeoDataFrame(data, geometry=geometries)


def _extract_carpark_dataframe(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> gpd.GeoDataFrame:

    ids = []
    left_boundaries = []
    right_boundaries = []
    outlines = []
    geometries = []

    for lane_helper in lane_helper_dict.values():
        if lane_helper.type == "parking":
            ids.append(lane_helper.lane_id)
            left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
            right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
            outlines.append(shapely.LineString(lane_helper.outline_polyline_3d))
            geometries.append(lane_helper.shapely_polygon)

    data = pd.DataFrame(
        {
            "id": ids,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
            "outline": outlines,
        }
    )
    return gpd.GeoDataFrame(data, geometry=geometries)


def _extract_generic_drivable_dataframe(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> gpd.GeoDataFrame:

    ids = []
    left_boundaries = []
    right_boundaries = []
    outlines = []
    geometries = []

    for lane_helper in lane_helper_dict.values():
        if lane_helper.type in ["none", "border", "bidirectional"]:
            ids.append(lane_helper.lane_id)
            left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
            right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
            outlines.append(shapely.LineString(lane_helper.outline_polyline_3d))
            geometries.append(lane_helper.shapely_polygon)

    data = pd.DataFrame(
        {
            "id": ids,
            "left_boundary": left_boundaries,
            "right_boundary": left_boundaries,
            "outline": outlines,
        }
    )
    return gpd.GeoDataFrame(data, geometry=geometries)


def _extract_intersections_dataframe(
    junction_dict: Dict[str, Junction],
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper],
) -> gpd.GeoDataFrame:
    def _find_lane_group_helpers_with_junction_id(junction_id: int) -> List[OpenDriveLaneGroupHelper]:
        return [
            lane_group_helper
            for lane_group_helper in lane_group_helper_dict.values()
            if lane_group_helper.junction_id == junction_id
        ]

    ids = []
    lane_group_ids = []
    geometries = []
    for junction in junction_dict.values():
        lane_group_helpers = _find_lane_group_helpers_with_junction_id(junction.id)
        lane_group_ids_ = [lane_group_helper.lane_group_id for lane_group_helper in lane_group_helpers]
        if len(lane_group_ids_) == 0:
            logger.debug(f"Skipping Junction {junction.id} without lane groups!")
            continue

        polygon = extract_exteriors_polygon(lane_group_helpers)
        ids.append(junction.id)
        lane_group_ids.append(lane_group_ids_)
        geometries.append(polygon)

    data = pd.DataFrame({"id": ids, "lane_group_ids": lane_group_ids})
    return gpd.GeoDataFrame(data, geometry=geometries)


def _extract_crosswalk_dataframe(object_helper_dict: Dict[int, OpenDriveObjectHelper]) -> gpd.GeoDataFrame:
    ids = []
    outlines = []
    geometries = []
    for object_helper in object_helper_dict.values():
        ids.append(object_helper.object_id)
        outlines.append(shapely.LineString(object_helper.outline_3d))
        geometries.append(object_helper.shapely_polygon)

    data = pd.DataFrame({"id": ids, "outline": outlines})
    return gpd.GeoDataFrame(data, geometry=geometries)


def _convert_ids_to_int(
    lane_df: gpd.GeoDataFrame,
    walkways_df: gpd.GeoDataFrame,
    carpark_df: gpd.GeoDataFrame,
    generic_drivable_area_df: gpd.GeoDataFrame,
    lane_group_df: gpd.GeoDataFrame,
    intersections_df: gpd.GeoDataFrame,
    crosswalk_df: gpd.GeoDataFrame,
) -> None:

    # NOTE: intersection and crosswalk ids are already integers

    # initialize id mappings
    lane_id_mapping = IntIDMapping.from_series(lane_df["id"])
    walkway_id_mapping = IntIDMapping.from_series(walkways_df["id"])
    carpark_id_mapping = IntIDMapping.from_series(carpark_df["id"])
    generic_drivable_id_mapping = IntIDMapping.from_series(generic_drivable_area_df["id"])
    lane_group_id_mapping = IntIDMapping.from_series(lane_group_df["id"])

    # Adjust cross reference in lane_df and lane_group_df
    lane_df["lane_group_id"] = lane_df["lane_group_id"].map(lane_group_id_mapping.str_to_int)
    lane_group_df["lane_ids"] = lane_group_df["lane_ids"].apply(lambda x: lane_id_mapping.map_list(x))

    # Adjust predecessor/successor in lane_df and lane_group_df
    for column in ["predecessor_ids", "successor_ids"]:
        lane_df[column] = lane_df[column].apply(lambda x: lane_id_mapping.map_list(x))
        lane_group_df[column] = lane_group_df[column].apply(lambda x: lane_group_id_mapping.map_list(x))

    for column in ["left_lane_id", "right_lane_id"]:
        lane_df[column] = lane_df[column].apply(
            lambda x: str(lane_id_mapping.str_to_int[x]) if pd.notna(x) and x is not None else x
        )

    lane_df["id"] = lane_df["id"].map(lane_id_mapping.str_to_int)
    walkways_df["id"] = walkways_df["id"].map(walkway_id_mapping.str_to_int)
    carpark_df["id"] = carpark_df["id"].map(carpark_id_mapping.str_to_int)
    generic_drivable_area_df["id"] = generic_drivable_area_df["id"].map(generic_drivable_id_mapping.str_to_int)
    lane_group_df["id"] = lane_group_df["id"].map(lane_group_id_mapping.str_to_int)

    intersections_df["lane_group_ids"] = intersections_df["lane_group_ids"].apply(
        lambda x: lane_group_id_mapping.map_list(x)
    )


def _extract_road_line_df(
    lane_df: gpd.GeoDataFrame,
    lane_group_df: gpd.GeoDataFrame,
) -> None:

    lane_group_on_intersection = {
        lane_group_id: str(intersection_id) != "nan"
        for lane_group_id, intersection_id in zip(lane_group_df.id.tolist(), lane_group_df.intersection_id.tolist())
    }
    ids = []
    road_line_types = []
    geometries = []

    running_id = 0
    for lane_row in lane_df.itertuples():
        on_intersection = lane_group_on_intersection.get(lane_row.lane_group_id, False)
        if on_intersection:
            # Skip road lines on intersections
            continue
        if str(lane_row.right_lane_id) in ["nan", "None"]:
            # This is a boundary lane, e.g. a border or sidewalk
            ids.append(running_id)
            road_line_types.append(int(RoadLineType.SOLID_WHITE))
            geometries.append(lane_row.right_boundary)
            running_id += 1
        else:
            # This is a regular lane
            ids.append(running_id)
            road_line_types.append(int(RoadLineType.DASHED_WHITE))
            geometries.append(lane_row.right_boundary)
            running_id += 1
        if str(lane_row.left_lane_id) in ["nan", "None"]:
            # This is a boundary lane, e.g. a border or sidewalk
            ids.append(running_id)
            road_line_types.append(int(RoadLineType.DASHED_WHITE))
            geometries.append(lane_row.left_boundary)
            running_id += 1

    data = pd.DataFrame({"id": ids, "road_line_type": road_line_types})
    return gpd.GeoDataFrame(data, geometry=geometries)


def _extract_road_edge_df(
    lane_df: gpd.GeoDataFrame,
    carpark_df: gpd.GeoDataFrame,
    generic_drivable_area_df: gpd.GeoDataFrame,
    lane_group_df: gpd.GeoDataFrame,
) -> None:
    road_edges = get_road_edges_3d_from_gdf(lane_df, carpark_df, generic_drivable_area_df, lane_group_df)
    road_edges = split_line_geometry_by_max_length(road_edges, MAX_ROAD_EDGE_LENGTH)

    ids = np.arange(len(road_edges), dtype=np.int64).tolist()
    # TODO @DanielDauner: Figure out if other types should/could be assigned here.
    road_edge_types = [int(RoadEdgeType.ROAD_EDGE_BOUNDARY)] * len(road_edges)
    geometries = road_edges
    return gpd.GeoDataFrame(pd.DataFrame({"id": ids, "road_edge_type": road_edge_types}), geometry=geometries)


# TODO: move this somewhere else and improve
def extract_exteriors_polygon(lane_group_helpers: List[OpenDriveLaneGroupHelper]) -> shapely.Polygon:

    # Step 1: Extract all boundary line segments
    all_polygons = []
    for lane_group_helper in lane_group_helpers:
        all_polygons.append(lane_group_helper.shapely_polygon)

    # Step 2: Merge all boundaries and extract the enclosed polygons
    # try:
    merged_boundaries = unary_union(all_polygons)
    # except Exception as e:
    #     warnings.warn(f"Topological error during polygon union: {e}")
    #     print([(helper.lane_group_id, poly.is_valid) for poly, helper in zip(all_polygons, lane_group_helpers)])
    #     merged_boundaries = unary_union([poly for poly in all_polygons if poly.is_valid])

    # Step 3: Generate polygons from the merged lines
    polygons = list(polygonize(merged_boundaries))

    # Step 4: Select the polygon that represents the intersection
    # Usually it's the largest polygon
    if len(polygons) == 1:
        return polygons[0]
    else:
        # Take the largest polygon if there are multiple
        return max(polygons, key=lambda p: p.area)
