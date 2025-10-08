from typing import List, Tuple

import pyarrow as pa

from d123.datatypes.vehicle_state.ego_state import EgoStateSE3Index
from d123.geometry.geometry_index import BoundingBoxSE3Index, Vector3DIndex

schema_list: List[Tuple[str, pa.DataType]] = [
    ("token", pa.string()),
    ("timestamp", pa.int64()),
]


def get_default_arrow_schema() -> List[Tuple[str, pa.DataType]]:
    return schema_list


def add_detection_schema(schema_list: List[Tuple[str, pa.DataType]]) -> None:
    detection_schema_addon: List[Tuple[str, pa.DataType]] = [
        ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
        ("detections_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
        ("detections_token", pa.list_(pa.string())),
        ("detections_type", pa.list_(pa.int16())),
    ]
    schema_list.extend(detection_schema_addon)


def add_traffic_light_schema(schema_list: List[Tuple[str, pa.DataType]]) -> None:
    traffic_light_schema_addon: List[Tuple[str, pa.DataType]] = [
        ("traffic_light_ids", pa.list_(pa.int64())),
        ("traffic_light_types", pa.list_(pa.int16())),
    ]
    schema_list.extend(traffic_light_schema_addon)


def add_ego_state_schema(schema_list: List[Tuple[str, pa.DataType]]) -> None:
    ego_state_schema_addon: List[Tuple[str, pa.DataType]] = [
        ("ego_state", pa.list_(pa.float64(), len(EgoStateSE3Index))),
    ]
    schema_list.extend(ego_state_schema_addon)


def add_route_schema(schema_list: List[Tuple[str, pa.DataType]]) -> None:
    route_schema_addon: List[Tuple[str, pa.DataType]] = [
        ("route_lane_group_ids", pa.list_(pa.int64())),
    ]
    add_route_schema(route_schema_addon)


def add_scenario_tags_schema(schema_list: List[Tuple[str, pa.DataType]]) -> None:
    scenario_tags_schema_addon: List[Tuple[str, pa.DataType]] = [
        ("scenario_tag", pa.list_(pa.string())),
    ]
    schema_list.extend(scenario_tags_schema_addon)
