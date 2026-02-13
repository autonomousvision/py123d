import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from py123d.conversion.utils.map_utils.opendrive.utils.lane_helper import OpenDriveLaneHelper
from py123d.conversion.utils.map_utils.opendrive.utils.signal_helper import OpenDriveSignalHelper
from py123d.datatypes.map_objects import StopZone, StopZoneType
from py123d.geometry.polyline import Polyline3D

logger = logging.getLogger(__name__)

STOP_ZONE_DEPTH = 0.5

# Signal reference key type: (road_id, signal_id, turn_relation)
SignalRefKey = Tuple[int, int, Optional[str]]


def _group_contiguous_lanes(
    helpers: List[OpenDriveLaneHelper],
) -> List[List[OpenDriveLaneHelper]]:
    """Group lane helpers into contiguous sequences by lane index.

    Lanes are first split by side (left/right), then grouped by contiguity
    within each side. Lanes are contiguous if their |id| values form an
    unbroken sequence.
    """
    if not helpers:
        return []

    left = sorted([h for h in helpers if h.id > 0], key=lambda h: h.id)
    right = sorted([h for h in helpers if h.id < 0], key=lambda h: abs(h.id))

    groups: List[List[OpenDriveLaneHelper]] = []
    for side_helpers in [left, right]:
        if not side_helpers:
            continue
        current_group: List[OpenDriveLaneHelper] = [side_helpers[0]]
        for i in range(1, len(side_helpers)):
            if abs(side_helpers[i].id) == abs(side_helpers[i - 1].id) + 1:
                current_group.append(side_helpers[i])
            else:
                groups.append(current_group)
                current_group = [side_helpers[i]]
        groups.append(current_group)
    return groups


SIGNAL_TYPE_MAP = {
    "1000001": StopZoneType.TRAFFIC_LIGHT,
    "206": StopZoneType.STOP_SIGN,
    "205": StopZoneType.YIELD_SIGN,
}


def _signal_type_to_stop_zone_type(signal: OpenDriveSignalHelper) -> StopZoneType:
    return SIGNAL_TYPE_MAP.get(signal.xodr_signal.type, StopZoneType.UNKNOWN)


def _create_stop_zone_outline_from_helpers(
    helpers: List[OpenDriveLaneHelper],
) -> Optional[Polyline3D]:
    """Create stop zone polygon from contiguous lane helpers.

    Places the stop zone at the beginning of the controlled lane.

    :param helpers: List of lane helpers (must be sorted by abs(id), contiguous)
    :return: Closed Polyline3D outline or None if no helpers
    """
    if not helpers:
        return None

    inner_helper = helpers[0]
    outer_helper = helpers[-1]

    # Place at lane beginning: right lanes travel in +s, left lanes in -s
    travels_in_s = inner_helper.id < 0
    if travels_in_s:
        start_s = inner_helper.s_range[0]
        end_s = start_s + STOP_ZONE_DEPTH
    else:
        end_s = inner_helper.s_range[1]
        start_s = end_s - STOP_ZONE_DEPTH

    start_s = np.clip(start_s, inner_helper.s_range[0], inner_helper.s_range[1])
    end_s = np.clip(end_s, inner_helper.s_range[0], inner_helper.s_range[1])
    outer_start_s = np.clip(start_s, outer_helper.s_range[0], outer_helper.s_range[1])
    outer_end_s = np.clip(end_s, outer_helper.s_range[0], outer_helper.s_range[1])

    inner_start = inner_helper.inner_boundary.interpolate_3d(start_s - inner_helper.s_range[0])
    outer_start = outer_helper.outer_boundary.interpolate_3d(outer_start_s - outer_helper.s_range[0])
    inner_end = inner_helper.inner_boundary.interpolate_3d(end_s - inner_helper.s_range[0])
    outer_end = outer_helper.outer_boundary.interpolate_3d(outer_end_s - outer_helper.s_range[0])

    corners = np.array(
        [
            inner_start,
            outer_start,
            outer_end,
            inner_end,
            inner_start,
        ]
    )

    return Polyline3D.from_array(corners)


def create_stop_zones_from_signals(
    signal_dict: Dict[SignalRefKey, OpenDriveSignalHelper],
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
) -> Dict[Tuple[int, int, Optional[str], int], StopZone]:
    """Create StopZone objects from signal reference helpers using lane geometry.

    Non-contiguous lanes are split into separate stop zones.

    :param signal_dict: Dictionary of signal helpers keyed by (road_id, signal_id, turn_relation)
    :param lane_helper_dict: Dictionary of lane helpers keyed by lane ID
    :return: Dictionary of StopZone objects keyed by (road_id, signal_id, turn_relation, group_idx)
    """
    stop_zones: Dict[Tuple[int, int, Optional[str], int], StopZone] = {}

    for key, signal_helper in signal_dict.items():
        stop_zone_type = _signal_type_to_stop_zone_type(signal_helper)
        if stop_zone_type == StopZoneType.UNKNOWN:
            continue

        # Skip signals without valid lane_ids
        if not signal_helper.lane_ids:
            continue

        # Get helpers for valid lane_ids
        helpers = [lane_helper_dict[lid] for lid in signal_helper.lane_ids if lid in lane_helper_dict]
        if not helpers:
            continue

        # Group into contiguous lane sequences
        groups = _group_contiguous_lanes(helpers)

        if len(groups) > 1:
            logger.debug(f"Signal {key} has non-contiguous lanes, creating {len(groups)} stop zones")

        for group_idx, group_helpers in enumerate(groups):
            outline = _create_stop_zone_outline_from_helpers(
                helpers=group_helpers,
            )

            if outline is None:
                continue

            # Generate unique object_id from extended key
            extended_key = (*key, group_idx)
            # Use MD5 for deterministic ID generation
            key_str = str(extended_key)
            object_id = int(hashlib.md5(key_str.encode("utf-8")).hexdigest(), 16) & 0x7FFFFFFF

            group_lane_ids = [h.lane_id for h in group_helpers]
            stop_zone = StopZone(
                object_id=object_id,
                stop_zone_type=stop_zone_type,
                outline=outline,
                lane_ids=group_lane_ids,
            )
            stop_zones[extended_key] = stop_zone

    return stop_zones
