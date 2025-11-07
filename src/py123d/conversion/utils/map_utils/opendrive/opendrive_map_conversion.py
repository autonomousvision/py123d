import logging
from pathlib import Path
from typing import Dict, Final, List

import numpy as np
import shapely

from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.utils.map_utils.opendrive.parser.opendrive import Junction, OpenDrive
from py123d.conversion.utils.map_utils.opendrive.utils.collection import collect_element_helpers
from py123d.conversion.utils.map_utils.opendrive.utils.lane_helper import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
)
from py123d.conversion.utils.map_utils.opendrive.utils.objects_helper import OpenDriveObjectHelper
from py123d.conversion.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)
from py123d.conversion.utils.map_utils.road_edge.road_edge_3d_utils import (
    get_road_edges_3d_from_drivable_surfaces,
    lift_outlines_to_3d,
)
from py123d.datatypes.map.cache.cache_map_objects import (
    CacheCarpark,
    CacheCrosswalk,
    CacheGenericDrivable,
    CacheIntersection,
    CacheLane,
    CacheLaneGroup,
    CacheRoadEdge,
    CacheRoadLine,
    CacheWalkway,
)
from py123d.datatypes.map.map_datatypes import RoadEdgeType, RoadLineType
from py123d.geometry.geometry_index import Point3DIndex
from py123d.geometry.polyline import Polyline3D

logger = logging.getLogger(__name__)

MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]


def convert_xodr_map(
    xordr_file: Path,
    map_writer: AbstractMapWriter,
    interpolation_step_size: float = 1.0,
    connection_distance_threshold: float = 0.1,
) -> None:

    opendrive = OpenDrive.parse_from_file(xordr_file)

    _, junction_dict, lane_helper_dict, lane_group_helper_dict, object_helper_dict = collect_element_helpers(
        opendrive, interpolation_step_size, connection_distance_threshold
    )

    # Collect data frames and store (needed for road edge/line extraction)
    lanes = _extract_and_write_lanes(lane_group_helper_dict, map_writer)
    lane_groups = _extract_and_write_lane_groups(lane_group_helper_dict, map_writer)
    car_parks = _extract_and_write_carparks(lane_helper_dict, map_writer)
    generic_drivables = _extract_and_write_generic_drivables(lane_helper_dict, map_writer)

    # Write other map elements
    _write_walkways(lane_helper_dict, map_writer)
    _write_intersections(junction_dict, lane_group_helper_dict, map_writer)
    _write_crosswalks(object_helper_dict, map_writer)

    # Extract polyline elements that are inferred of other road surfaces.
    _write_road_lines(lanes, lane_groups, map_writer)
    _write_road_edges(
        lanes=lanes,
        lane_groups=lane_groups,
        car_parks=car_parks,
        generic_drivables=generic_drivables,
        map_writer=map_writer,
    )


def _extract_and_write_lanes(
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper],
    map_writer: AbstractMapWriter,
) -> List[CacheLane]:

    lanes: List[CacheLane] = []
    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_id = lane_group_helper.lane_group_id
        lane_helpers = lane_group_helper.lane_helpers
        num_lanes = len(lane_helpers)
        # NOTE: Lanes are going left to right, ie. inner to outer
        for lane_idx, lane_helper in enumerate(lane_helpers):
            left_lane_id = lane_helpers[lane_idx - 1].lane_id if lane_idx > 0 else None
            right_lane_id = lane_helpers[lane_idx + 1].lane_id if lane_idx < num_lanes - 1 else None
            lane = CacheLane(
                object_id=lane_helper.lane_id,
                lane_group_id=lane_group_id,
                left_boundary=lane_helper.inner_polyline_3d,
                right_boundary=lane_helper.outer_polyline_3d,
                centerline=lane_helper.center_polyline_3d,
                left_lane_id=left_lane_id,
                right_lane_id=right_lane_id,
                predecessor_ids=lane_helper.predecessor_lane_ids,
                successor_ids=lane_helper.successor_lane_ids,
                speed_limit_mps=lane_helper.speed_limit_mps,
                outline=lane_helper.outline_polyline_3d,
                geometry=None,
            )
            lanes.append(lane)
            map_writer.write_lane(lane)

    return lanes


def _extract_and_write_lane_groups(
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper], map_writer: AbstractMapWriter
) -> List[CacheLaneGroup]:

    lane_groups: List[CacheLaneGroup] = []
    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_helper: OpenDriveLaneGroupHelper
        lane_group = CacheLaneGroup(
            object_id=lane_group_helper.lane_group_id,
            lane_ids=[lane_helper.lane_id for lane_helper in lane_group_helper.lane_helpers],
            left_boundary=lane_group_helper.inner_polyline_3d,
            right_boundary=lane_group_helper.outer_polyline_3d,
            intersection_id=lane_group_helper.junction_id,
            predecessor_ids=lane_group_helper.predecessor_lane_group_ids,
            successor_ids=lane_group_helper.successor_lane_group_ids,
            outline=lane_group_helper.outline_polyline_3d,
            geometry=None,
        )
        lane_groups.append(lane_group)
        map_writer.write_lane_group(lane_group)

    return lane_groups


def _write_walkways(lane_helper_dict: Dict[str, OpenDriveLaneHelper], map_writer: AbstractMapWriter) -> None:
    for lane_helper in lane_helper_dict.values():
        if lane_helper.type == "sidewalk":
            map_writer.write_walkway(
                CacheWalkway(
                    object_id=lane_helper.lane_id,
                    outline=lane_helper.outline_polyline_3d,
                    geometry=None,
                )
            )


def _extract_and_write_carparks(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper], map_writer: AbstractMapWriter
) -> List[CacheCarpark]:

    carparks: List[CacheCarpark] = []
    for lane_helper in lane_helper_dict.values():
        if lane_helper.type == "parking":
            carpark = CacheCarpark(
                object_id=lane_helper.lane_id,
                outline=lane_helper.outline_polyline_3d,
                geometry=None,
            )
            carparks.append(carpark)
            map_writer.write_carpark(carpark)

    return carparks


def _extract_and_write_generic_drivables(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper], map_writer: AbstractMapWriter
) -> List[CacheGenericDrivable]:

    generic_drivables: List[CacheGenericDrivable] = []
    for lane_helper in lane_helper_dict.values():
        if lane_helper.type in ["none", "border", "bidirectional"]:
            generic_drivable = CacheGenericDrivable(
                object_id=lane_helper.lane_id,
                outline=lane_helper.outline_polyline_3d,
                geometry=None,
            )
            generic_drivables.append(generic_drivable)
            map_writer.write_generic_drivable(generic_drivable)
    return generic_drivables


def _write_intersections(
    junction_dict: Dict[str, Junction],
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper],
    map_writer: AbstractMapWriter,
) -> None:

    def _find_lane_group_helpers_with_junction_id(junction_id: int) -> List[OpenDriveLaneGroupHelper]:
        return [
            lane_group_helper
            for lane_group_helper in lane_group_helper_dict.values()
            if lane_group_helper.junction_id == junction_id
        ]

    for junction in junction_dict.values():
        lane_group_helpers = _find_lane_group_helpers_with_junction_id(junction.id)
        lane_group_ids_ = [lane_group_helper.lane_group_id for lane_group_helper in lane_group_helpers]
        if len(lane_group_ids_) == 0:
            logger.debug(f"Skipping Junction {junction.id} without lane groups!")
            continue

        # TODO @DanielDauner: Create a method that extracts 3D outlines of intersections.
        outline = _extract_intersection_outline(lane_group_helpers, junction.id)
        map_writer.write_intersection(
            CacheIntersection(
                object_id=junction.id,
                lane_group_ids=lane_group_ids_,
                outline=outline,
                geometry=None,
            )
        )


def _write_crosswalks(object_helper_dict: Dict[int, OpenDriveObjectHelper], map_writer: AbstractMapWriter) -> None:
    for object_helper in object_helper_dict.values():
        map_writer.write_crosswalk(
            CacheCrosswalk(
                object_id=object_helper.object_id,
                outline=object_helper.outline_polyline_3d,
                geometry=None,
            )
        )


def _write_road_lines(lanes: List[CacheLane], lane_groups: List[CacheLaneGroup], map_writer: AbstractMapWriter) -> None:

    # NOTE @DanielDauner: This method of extracting road lines is very simplistic and needs improvement.
    # The OpenDRIVE format provides lane boundary types that could be used here.
    # Additionally, the logic of inferring road lines is somewhat flawed, e.g, assuming constant types/colors of lines.

    lane_group_on_intersection: Dict[str, bool] = {
        lane_group.object_id: lane_group.intersection_id is not None for lane_group in lane_groups
    }

    ids: List[int] = []
    road_line_types: List[RoadLineType] = []
    polylines: List[Polyline3D] = []

    running_id = 0
    for lane in lanes:

        on_intersection = lane_group_on_intersection.get(lane.lane_group_id, False)
        if on_intersection:
            # Skip road lines on intersections
            continue

        if lane.right_lane_id is None:
            # This is a boundary lane, e.g. a border or sidewalk
            ids.append(running_id)
            road_line_types.append(RoadLineType.SOLID_WHITE)
            polylines.append(lane.right_boundary)
            running_id += 1
        else:
            # This is a regular lane
            ids.append(running_id)
            road_line_types.append(RoadLineType.DASHED_WHITE)
            polylines.append(lane.right_boundary)
            running_id += 1
        if lane.left_lane_id is None:
            # This is a boundary lane, e.g. a border or sidewalk
            ids.append(running_id)
            road_line_types.append(RoadLineType.DASHED_WHITE)
            polylines.append(lane.left_boundary)
            running_id += 1

    for object_id, road_line_type, polyline in zip(ids, road_line_types, polylines):
        map_writer.write_road_line(
            CacheRoadLine(
                object_id=object_id,
                road_line_type=road_line_type,
                polyline=polyline,
            )
        )


def _write_road_edges(
    lanes: List[CacheLane],
    lane_groups: List[CacheLaneGroup],
    car_parks: List[CacheCarpark],
    generic_drivables: List[CacheGenericDrivable],
    map_writer: AbstractMapWriter,
) -> None:

    road_edges_ = get_road_edges_3d_from_drivable_surfaces(
        lanes=lanes,
        lane_groups=lane_groups,
        car_parks=car_parks,
        generic_drivables=generic_drivables,
    )
    road_edge_linestrings = split_line_geometry_by_max_length(
        [road_edges.linestring for road_edges in road_edges_], MAX_ROAD_EDGE_LENGTH
    )

    running_id = 0
    for road_edge_linestring in road_edge_linestrings:
        #  TODO @DanielDauner: Figure out if other types should/could be assigned here.
        map_writer.write_road_edge(
            CacheRoadEdge(
                object_id=running_id,
                road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                polyline=Polyline3D.from_linestring(road_edge_linestring),
            )
        )
        running_id += 1


def _extract_intersection_outline(lane_group_helpers: List[OpenDriveLaneGroupHelper], junction_id: str) -> Polyline3D:
    """Helper method to extract intersection outline in 3D from lane group helpers."""

    # 1. Extract the intersection outlines in 2D
    intersection_polygons: List[shapely.Polygon] = [
        lane_group_helper.shapely_polygon for lane_group_helper in lane_group_helpers
    ]
    intersection_edges = get_road_edge_linear_rings(
        intersection_polygons,
        buffer_distance=0.25,
        add_interiors=False,
    )

    # 2. Lift the 2D outlines to 3D
    lane_group_outlines: List[Polyline3D] = [
        lane_group_helper.outline_polyline_3d for lane_group_helper in lane_group_helpers
    ]
    intersection_outlines = lift_outlines_to_3d(intersection_edges, lane_group_outlines)

    # NOTE: When the intersection has multiple non-overlapping outlines, we cannot return a single outline in 3D.
    # For now, we return the longest outline.
    valid_outlines = [outline for outline in intersection_outlines if outline.array.shape[0] > 2]
    if len(valid_outlines) == 0:
        logging.warning(
            f"Could not extract valid outline for intersection {junction_id} with {len(intersection_edges)} edges!"
        )
        longest_outline_2d = max(intersection_edges, key=lambda outline: outline.length)
        average_z = sum(outline.array[:, 2].mean() for outline in intersection_outlines) / len(intersection_outlines)

        outline_3d_array = np.zeros((len(longest_outline_2d.coords), 3))
        outline_3d_array[:, Point3DIndex.XY] = np.array(longest_outline_2d.coords)
        outline_3d_array[:, Point3DIndex.Z] = average_z
        longest_outline = Polyline3D.from_array(outline_3d_array)
    else:
        longest_outline = max(valid_outlines, key=lambda outline: outline.length)

    return longest_outline
