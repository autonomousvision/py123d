import pyarrow as pa

from d123.datatypes.vehicle_state.ego_state import EgoStateSE3Index
from d123.geometry.geometry_index import BoundingBoxSE3Index, Vector3DIndex

schema_column_list = [
    ("token", pa.string()),
    ("timestamp", pa.int64()),
    ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
    ("detections_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
    ("detections_token", pa.list_(pa.string())),
    ("detections_type", pa.list_(pa.int16())),
    ("ego_states", pa.list_(pa.float64(), len(EgoStateSE3Index))),
    ("traffic_light_ids", pa.list_(pa.int64())),
    ("traffic_light_types", pa.list_(pa.int16())),
    ("scenario_tag", pa.list_(pa.string())),
    ("route_lane_group_ids", pa.list_(pa.int64())),
]


def get_default_arrow_schema() -> pa.schema:
    return pa.schema(schema_column_list)
