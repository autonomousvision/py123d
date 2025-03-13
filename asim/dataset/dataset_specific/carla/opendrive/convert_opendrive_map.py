from typing import Dict, List

from asim.dataset.dataset_specific.carla.opendrive.conversion.group_collections import (
    ODLaneGroupHelper,
    lane_section_to_lane_group_helper,
)
from asim.dataset.dataset_specific.carla.opendrive.elements.opendrive import OpenDRIVE
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import Border


def convert_opendrive_map(opendrive: OpenDRIVE) -> Dict[str, List[ODLaneGroupHelper]]:
    road_lane_groups: Dict[str, List[ODLaneGroupHelper]] = {}

    for road in opendrive.roads:
        road_lane_groups[road.id] = []

        reference_border = Border.from_plan_view(road.plan_view, road.lanes.lane_offsets)
        lane_section_lengths: List[float] = [ls.s for ls in road.lanes.lane_sections] + [road.length]

        for idx, lane_section in enumerate(road.lanes.lane_sections):
            parametric_lanes_ = lane_section_to_lane_group_helper(
                lane_section,
                reference_border,
                lane_section_lengths[idx],
                lane_section_lengths[idx + 1],
            )
            road_lane_groups[road.id].extend(parametric_lanes_)

    return road_lane_groups
