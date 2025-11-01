import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from py123d.conversion.utils.map_utils.opendrive.parser.opendrive import Junction, OpenDrive
from py123d.conversion.utils.map_utils.opendrive.parser.reference import ReferenceLine
from py123d.conversion.utils.map_utils.opendrive.parser.road import Road
from py123d.conversion.utils.map_utils.opendrive.utils.id_system import (
    build_lane_id,
    derive_lane_section_id,
    lane_group_id_from_lane_id,
    road_id_from_lane_group_id,
)
from py123d.conversion.utils.map_utils.opendrive.utils.lane_helper import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
    lane_section_to_lane_helpers,
)
from py123d.conversion.utils.map_utils.opendrive.utils.objects_helper import OpenDriveObjectHelper, get_object_helper

logger = logging.getLogger(__name__)


def collect_element_helpers(
    opendrive: OpenDrive,
    interpolation_step_size: float,
    connection_distance_threshold: float,
) -> Tuple[
    Dict[int, Road],
    Dict[int, Junction],
    Dict[str, OpenDriveLaneHelper],
    Dict[str, OpenDriveLaneGroupHelper],
    Dict[int, OpenDriveObjectHelper],
]:

    # 1. Fill the road and junction dictionaries
    road_dict: Dict[int, Road] = {road.id: road for road in opendrive.roads}
    junction_dict: Dict[int, Junction] = {junction.id: junction for junction in opendrive.junctions}

    # 2. Create lane helpers from the roads
    lane_helper_dict: Dict[str, OpenDriveLaneHelper] = {}
    for road in opendrive.roads:
        reference_line = ReferenceLine.from_plan_view(
            road.plan_view,
            road.lanes.lane_offsets,
            road.elevation_profile.elevations,
        )
        lane_section_lengths: List[float] = [ls.s for ls in road.lanes.lane_sections] + [road.length]
        for idx, lane_section in enumerate(road.lanes.lane_sections):
            lane_section_id = derive_lane_section_id(road.id, idx)
            lane_helpers_ = lane_section_to_lane_helpers(
                lane_section_id,
                lane_section,
                reference_line,
                lane_section_lengths[idx],
                lane_section_lengths[idx + 1],
                road.road_types,
                interpolation_step_size,
            )
            lane_helper_dict.update(lane_helpers_)

    # 3. Update the connections and fill the lane helpers:
    # 3.1. From links of the roads
    _update_connection_from_links(lane_helper_dict, road_dict)
    # 3.2. From junctions
    _update_connection_from_junctions(lane_helper_dict, junction_dict, road_dict)
    # 3.3. Flip the connections to align to the lane direction
    _flip_and_set_connections(lane_helper_dict)
    # 3.4. Remove invalid connections based on centerline distances
    _post_process_connections(lane_helper_dict, connection_distance_threshold)

    # 4. Collect lane groups from lane helpers
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper] = _collect_lane_groups(
        lane_helper_dict, junction_dict, road_dict
    )

    # 5. Collect objects, i.e. crosswalks
    crosswalk_dict = _collect_crosswalks(opendrive)

    return (road_dict, junction_dict, lane_helper_dict, lane_group_helper_dict, crosswalk_dict)


def _update_connection_from_links(lane_helper_dict: Dict[str, OpenDriveLaneHelper], road_dict: Dict[int, Road]) -> None:
    """
    Uses the links of the roads to update the connections between lane helpers.
    :param lane_helper_dict: Dictionary of lane helpers indexed by lane id.
    :param road_dict: Dictionary of roads indexed by road id.
    """

    for lane_id in lane_helper_dict.keys():
        road_idx, lane_section_idx, _, lane_idx = lane_id.split("_")
        road_idx, lane_section_idx, lane_idx = int(road_idx), int(lane_section_idx), int(lane_idx)

        road = road_dict[road_idx]

        successor_lane_idx = lane_helper_dict[lane_id].open_drive_lane.successor
        if successor_lane_idx is not None:
            successor_lane_id: Optional[str] = None

            # Not the last lane section -> Next lane section in same road
            if lane_section_idx < road.lanes.last_lane_section_idx:
                successor_lane_id = build_lane_id(
                    road_idx,
                    lane_section_idx + 1,
                    successor_lane_idx,
                )

            # Last lane section -> Next road in first lane section
            # Try to get next road
            elif road.link.successor is not None and road.link.successor.element_type != "junction":
                successor_road = road_dict[road.link.successor.element_id]
                successor_lane_section_idx = (
                    0 if road.link.successor.contact_point == "start" else successor_road.lanes.last_lane_section_idx
                )

                successor_lane_id = build_lane_id(
                    successor_road.id,
                    successor_lane_section_idx,
                    successor_lane_idx,
                )

            # assert successor_lane_id in lane_helper_dict.keys()
            if successor_lane_id is None or successor_lane_id not in lane_helper_dict.keys():
                continue
            lane_helper_dict[lane_id].successor_lane_ids.append(successor_lane_id)
            lane_helper_dict[successor_lane_id].predecessor_lane_ids.append(lane_id)

        predecessor_lane_idx = lane_helper_dict[lane_id].open_drive_lane.predecessor
        if predecessor_lane_idx is not None:
            predecessor_lane_id: Optional[str] = None

            # Not the first lane section -> Previous lane section in same road
            if lane_section_idx > 0:
                predecessor_lane_id = build_lane_id(
                    road_idx,
                    lane_section_idx - 1,
                    predecessor_lane_idx,
                )

            # First lane section -> Previous road
            # Try to get previous road
            elif road.link.predecessor is not None and road.link.predecessor.element_type != "junction":
                predecessor_road = road_dict[road.link.predecessor.element_id]
                predecessor_lane_section_idx = (
                    0
                    if road.link.predecessor.contact_point == "start"
                    else predecessor_road.lanes.last_lane_section_idx
                )

                predecessor_lane_id = build_lane_id(
                    predecessor_road.id,
                    predecessor_lane_section_idx,
                    predecessor_lane_idx,
                )

            # assert predecessor_lane_id in lane_helper_dict.keys()
            if predecessor_lane_id is None or predecessor_lane_id not in lane_helper_dict.keys():
                continue
            lane_helper_dict[lane_id].predecessor_lane_ids.append(predecessor_lane_id)
            lane_helper_dict[predecessor_lane_id].successor_lane_ids.append(lane_id)


def _update_connection_from_junctions(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    junction_dict: Dict[int, Junction],
    road_dict: Dict[int, Road],
) -> None:
    """
    Helper function to update the lane connections based on junctions.
    :param lane_helper_dict: Dictionary mapping lane ids to their helper objects.
    :param junction_dict: Dictionary mapping junction ids to their objects.
    :param road_dict: Dictionary mapping road ids to their objects.
    :raises ValueError: If a connection is invalid.
    """

    for junction_idx, junction in junction_dict.items():
        for connection in junction.connections:
            # Skip connections with invalid/virtual road IDs (e.g., -1)
            # TODO : this requires further investigation
            # Is this the correct way to handle such cases or should a map be
            # complete with all roads defined?
            if connection.incoming_road not in road_dict or connection.connecting_road not in road_dict:
                logger.debug(
                    f"Skipping junction connection with incoming_road={connection.incoming_road}, "
                    f"connecting_road={connection.connecting_road} - road(s) not found in road_dict"
                )
                continue

            incoming_road = road_dict[connection.incoming_road]
            connecting_road = road_dict[connection.connecting_road]

            for lane_link in connection.lane_links:

                incoming_lane_id: Optional[str] = None
                connecting_lane_id: Optional[str] = None

                if connection.contact_point == "start":
                    incoming_lane_section_idx = incoming_road.lanes.last_lane_section_idx if lane_link.start < 0 else 0
                    incoming_lane_id = build_lane_id(incoming_road.id, incoming_lane_section_idx, lane_link.start)
                    connecting_lane_id = build_lane_id(connecting_road.id, 0, lane_link.end)
                elif connection.contact_point == "end":
                    incoming_lane_id = build_lane_id(incoming_road.id, 0, lane_link.start)
                    connecting_lane_id = build_lane_id(
                        connecting_road.id,
                        connecting_road.lanes.last_lane_section_idx,
                        lane_link.end,
                    )
                else:
                    raise ValueError(f"Unknown contact point: {connection.contact_point} in junction {junction_idx}")

                if incoming_lane_id is None or connecting_lane_id is None:
                    logger.debug(f"OpenDRIVE: Lane connection {incoming_lane_id} -> {connecting_lane_id} not valid")
                    continue
                if incoming_lane_id not in lane_helper_dict.keys() or connecting_lane_id not in lane_helper_dict.keys():
                    logger.debug(
                        f"OpenDRIVE: Lane connection {incoming_lane_id} -> {connecting_lane_id} not found in lane_helper_dict"
                    )
                    continue
                lane_helper_dict[incoming_lane_id].successor_lane_ids.append(connecting_lane_id)
                lane_helper_dict[connecting_lane_id].predecessor_lane_ids.append(incoming_lane_id)


def _flip_and_set_connections(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> None:
    """
    Helper function to flip the connections of the lane helpers, to align them with the lane direction
    :param lane_helper_dict: Dictionary mapping lane ids to their helper objects.
    """

    for lane_id in lane_helper_dict.keys():
        if lane_helper_dict[lane_id].id > 0:
            successors_temp = lane_helper_dict[lane_id].successor_lane_ids
            lane_helper_dict[lane_id].successor_lane_ids = lane_helper_dict[lane_id].predecessor_lane_ids
            lane_helper_dict[lane_id].predecessor_lane_ids = successors_temp
        lane_helper_dict[lane_id].successor_lane_ids = list(set(lane_helper_dict[lane_id].successor_lane_ids))
        lane_helper_dict[lane_id].predecessor_lane_ids = list(set(lane_helper_dict[lane_id].predecessor_lane_ids))


def _post_process_connections(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    connection_distance_threshold: float,
) -> None:
    """
    Helper function to post-process the connections of the lane helpers, removing invalid connections based on centerline distances.
    :param lane_helper_dict: Dictionary mapping lane ids to their helper objects.
    :param connection_distance_threshold: Threshold distance for valid connections.
    """

    for lane_id in lane_helper_dict.keys():
        lane_helper_dict[lane_id]

        centerline = lane_helper_dict[lane_id].center_polyline_se2

        valid_successor_lane_ids: List[str] = []
        for successor_lane_id in lane_helper_dict[lane_id].successor_lane_ids:
            successor_centerline = lane_helper_dict[successor_lane_id].center_polyline_se2
            distance = np.linalg.norm(centerline[-1, :2] - successor_centerline[0, :2])
            if distance > connection_distance_threshold:
                logger.debug(
                    f"OpenDRIVE: Removing connection {lane_id} -> {successor_lane_id} with distance {distance}"
                )
            else:
                valid_successor_lane_ids.append(successor_lane_id)
        lane_helper_dict[lane_id].successor_lane_ids = valid_successor_lane_ids

        valid_predecessor_lane_ids: List[str] = []
        for predecessor_lane_id in lane_helper_dict[lane_id].predecessor_lane_ids:
            predecessor_centerline = lane_helper_dict[predecessor_lane_id].center_polyline_se2
            distance = np.linalg.norm(centerline[0, :2] - predecessor_centerline[-1, :2])
            if distance > connection_distance_threshold:
                # Note: this caused some issues, had to change it to lane_id
                logger.debug(
                    f"OpenDRIVE: Removing connection {predecessor_lane_id} -> {lane_id} with distance {distance}"
                )
            else:
                valid_predecessor_lane_ids.append(predecessor_lane_id)
        lane_helper_dict[lane_id].predecessor_lane_ids = valid_predecessor_lane_ids


def _collect_lane_groups(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    junction_dict: Dict[int, Junction],
    road_dict: Dict[int, Road],
) -> None:

    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper] = {}

    def _collect_lane_helper_of_id(lane_group_id: str) -> List[OpenDriveLaneHelper]:
        lane_helpers: List[OpenDriveLaneHelper] = []
        for lane_id, lane_helper in lane_helper_dict.items():
            if (lane_helper.type in ["driving"]) and (lane_group_id_from_lane_id(lane_id) == lane_group_id):
                lane_helpers.append(lane_helper)
        return lane_helpers

    def _collect_lane_group_ids_of_road(road_id: int) -> List[str]:
        lane_group_ids: List[str] = []
        for lane_group_id in lane_group_helper_dict.keys():
            if int(road_id_from_lane_group_id(lane_group_id)) == road_id:
                lane_group_ids.append(lane_group_id)
        return lane_group_ids

    all_lane_group_ids = list(set([lane_group_id_from_lane_id(lane_id) for lane_id in lane_helper_dict.keys()]))

    for lane_group_id in all_lane_group_ids:
        lane_group_lane_helper = _collect_lane_helper_of_id(lane_group_id)
        if len(lane_group_lane_helper) >= 1:
            lane_group_helper_dict[lane_group_id] = OpenDriveLaneGroupHelper(lane_group_id, lane_group_lane_helper)

    for junction in junction_dict.values():
        for connection in junction.connections:
            connecting_road = road_dict[connection.connecting_road]
            connecting_lane_group_ids = _collect_lane_group_ids_of_road(connecting_road.id)
            for connecting_lane_group_id in connecting_lane_group_ids:
                if connecting_lane_group_id in lane_group_helper_dict.keys():
                    lane_group_helper_dict[connecting_lane_group_id].junction_id = junction.id

    return lane_group_helper_dict


def _collect_crosswalks(opendrive: OpenDrive) -> Dict[int, OpenDriveObjectHelper]:

    object_helper_dict: Dict[int, OpenDriveObjectHelper] = {}
    for road in opendrive.roads:
        if len(road.objects) == 0:
            continue
        reference_line = ReferenceLine.from_plan_view(
            road.plan_view,
            road.lanes.lane_offsets,
            road.elevation_profile.elevations,
        )
        for object in road.objects:
            if object.type in ["crosswalk"]:
                object_helper = get_object_helper(object, reference_line)
                object_helper_dict[object_helper.object_id] = object_helper

    return object_helper_dict
