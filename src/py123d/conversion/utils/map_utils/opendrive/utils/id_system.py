from typing import Union


def derive_lane_section_id(road_idx: Union[int, str], lane_section_idx: Union[int, str]) -> str:
    return f"{road_idx}_{lane_section_idx}"


def derive_lane_group_id(lane_section_id: Union[int, str], side: str) -> str:
    assert side in ["left", "center", "right"]
    return f"{lane_section_id}_{side}"


def derive_lane_id(lane_group_id: Union[int, str], lane_idx: Union[int, str]) -> str:
    return f"{lane_group_id}_{lane_idx}"


def build_lane_id(road_idx: Union[int, str], lane_section_idx: Union[int, str], lane_idx: Union[int, str]) -> str:
    side = "right" if int(lane_idx) < 0 else "left"
    return f"{road_idx}_{lane_section_idx}_{side}_{lane_idx}"


def build_lane_group_id(road_idx: Union[int, str], lane_section_idx: Union[int, str], side: str) -> str:
    return f"{road_idx}_{lane_section_idx}_{side}"


def lane_group_id_from_lane_id(lane_id: str) -> str:
    road_idx, lane_section_idx, side, _ = lane_id.split("_")
    return build_lane_group_id(road_idx, lane_section_idx, side)


def road_id_from_lane_id(lane_id: str) -> str:
    road_idx, lane_section_idx, side, _ = lane_id.split("_")
    return road_idx


def road_id_from_lane_group_id(lane_group_id: str) -> str:
    road_idx, lane_section_idx, side = lane_group_id.split("_")
    return road_idx


def lane_section_id_from_lane_group_id(lane_group_id: str) -> str:
    road_idx, lane_section_idx, side = lane_group_id.split("_")
    return f"{road_idx}_{lane_section_idx}"
