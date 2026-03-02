from typing import Callable, Final, List

import pyarrow as pa

from py123d.datatypes import DynamicStateSE3Index
from py123d.geometry import BoundingBoxSE3Index, PoseSE3Index
from py123d.geometry.geometry_index import Vector3DIndex

# Synchronization file
# ----------------------------------------------------------------------------------------------------------------------
SYNC_NAME: Final[str] = "sync"
SYNC_SCHEMA_DICT: Final[dict] = {
    f"{SYNC_NAME}.uuid": pa.uuid(),
    f"{SYNC_NAME}.timestamp_us": pa.int64(),
}

UUID_COLUMN: Final[str] = f"{SYNC_NAME}.uuid"
TIMESTAMP_US_COLUMN: Final[str] = f"{SYNC_NAME}.timestamp_us"

# EgoStateSE3 file
# ----------------------------------------------------------------------------------------------------------------------
EGO_STATE_SE3_NAME: Final[str] = "ego_state_se3"
EGO_STATE_SE3_SCHEMA_DICT: Final[dict] = {
    f"{EGO_STATE_SE3_NAME}.imu_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
    f"{EGO_STATE_SE3_NAME}.dynamic_state_se3": pa.list_(pa.float64(), len(DynamicStateSE3Index)),
    f"{EGO_STATE_SE3_NAME}.timestamp_us": pa.int64(),
}

EGO_IMU_SE3_COLUMN: Final[str] = f"{EGO_STATE_SE3_NAME}.imu_se3"
EGO_DYNAMIC_STATE_SE3_COLUMN: Final[str] = f"{EGO_STATE_SE3_NAME}.dynamic_state_se3"
EGO_TIMESTAMP_US_COLUMN: Final[str] = f"{EGO_STATE_SE3_NAME}.timestamp_us"
EGO_STATE_SE3_COLUMNS: Final[List[str]] = [EGO_IMU_SE3_COLUMN, EGO_DYNAMIC_STATE_SE3_COLUMN, EGO_TIMESTAMP_US_COLUMN]


# BoxDetectionsSE3 file
# ----------------------------------------------------------------------------------------------------------------------
BOX_DETECTIONS_SE3_NAME: Final[str] = "box_detections_se3"
BOX_DETECTIONS_SE3_SCHEMA_DICT: Final[dict] = {
    f"{BOX_DETECTIONS_SE3_NAME}.bounding_box_se3": pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index))),
    f"{BOX_DETECTIONS_SE3_NAME}.token": pa.list_(pa.string()),
    f"{BOX_DETECTIONS_SE3_NAME}.label": pa.list_(pa.uint16()),
    f"{BOX_DETECTIONS_SE3_NAME}.velocity_3d": pa.list_(pa.list_(pa.float64(), len(Vector3DIndex))),
    f"{BOX_DETECTIONS_SE3_NAME}.num_lidar_points": pa.list_(pa.int32()),
}

BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN: Final[str] = f"{BOX_DETECTIONS_SE3_NAME}.bounding_box_se3"
BOX_DETECTIONS_TOKEN_COLUMN: Final[str] = f"{BOX_DETECTIONS_SE3_NAME}.token"
BOX_DETECTIONS_LABEL_COLUMN: Final[str] = f"{BOX_DETECTIONS_SE3_NAME}.label"
BOX_DETECTIONS_VELOCITY_3D_COLUMN: Final[str] = f"{BOX_DETECTIONS_SE3_NAME}.velocity_3d"
BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN: Final[str] = f"{BOX_DETECTIONS_SE3_NAME}.num_lidar_points"
BOX_DETECTIONS_SE3_COLUMNS: Final[List[str]] = [
    BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN,
    BOX_DETECTIONS_TOKEN_COLUMN,
    BOX_DETECTIONS_LABEL_COLUMN,
    BOX_DETECTIONS_VELOCITY_3D_COLUMN,
    BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN,
]

# TrafficLights file
# ----------------------------------------------------------------------------------------------------------------------
TRAFFIC_LIGHTS_NAME: Final[str] = "traffic_lights"
TRAFFIC_LIGHTS_SCHEMA_DICT: Final[dict] = {
    f"{TRAFFIC_LIGHTS_NAME}.lane_id": pa.int32(),
    f"{TRAFFIC_LIGHTS_NAME}.status": pa.uint8(),
    f"{TRAFFIC_LIGHTS_NAME}.timestamp_us": pa.int64(),
}

TRAFFIC_LIGHTS_LANE_ID_COLUMN: Final[str] = f"{TRAFFIC_LIGHTS_NAME}.lane_id"
TRAFFIC_LIGHTS_STATUS_COLUMN: Final[str] = f"{TRAFFIC_LIGHTS_NAME}.status"
TRAFFIC_LIGHTS_TIMESTAMP_US_COLUMN: Final[str] = f"{TRAFFIC_LIGHTS_NAME}.timestamp_us"
TRAFFIC_LIGHTS_COLUMNS: Final[List[str]] = [
    TRAFFIC_LIGHTS_LANE_ID_COLUMN,
    TRAFFIC_LIGHTS_STATUS_COLUMN,
    TRAFFIC_LIGHTS_TIMESTAMP_US_COLUMN,
]

# PinholeCamera file
# ----------------------------------------------------------------------------------------------------------------------
PCAM_NAME: Callable[[str], str] = lambda name: f"pinhole_camera.{name}"

PCAM_STRING_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{PCAM_NAME(name)}.data": pa.string(),
    f"{PCAM_NAME(name)}.state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
    f"{PCAM_NAME(name)}.timestamp_us": pa.int64(),
}

PCAM_BINARY_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{PCAM_NAME(name)}.data": pa.binary(),
    f"{PCAM_NAME(name)}.state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
    f"{PCAM_NAME(name)}.timestamp_us": pa.int64(),
}


PCAM_INT_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{PCAM_NAME(name)}.data": pa.int64(),
    f"{PCAM_NAME(name)}.state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
    f"{PCAM_NAME(name)}.timestamp_us": pa.int64(),
}

PINHOLE_CAMERA_DATA_COLUMN: Callable[[str], str] = lambda name: f"{PCAM_NAME(name)}.data"
PINHOLE_CAMERA_EXTRINSIC_COLUMN: Callable[[str], str] = lambda name: f"{PCAM_NAME(name)}.state_se3"
PINHOLE_CAMERA_TIMESTAMP_COLUMN: Callable[[str], str] = lambda name: f"{PCAM_NAME(name)}.timestamp_us"


# Fisheye MEI Cameras
# ----------------------------------------------------------------------------------------------------------------------
FCAM_NAME: Callable[[str], str] = lambda name: f"fisheye_mei.{name}"

FCAM_STRING_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{FCAM_NAME(name)}.data": pa.string(),
    f"{FCAM_NAME(name)}.state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
    f"{FCAM_NAME(name)}.timestamp_us": pa.int64(),
}
FCAM_BINARY_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{FCAM_NAME(name)}.data": pa.binary(),
    f"{FCAM_NAME(name)}.state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
    f"{FCAM_NAME(name)}.timestamp_us": pa.int64(),
}
FCAM_INT_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{FCAM_NAME(name)}.data": pa.int64(),
    f"{FCAM_NAME(name)}.state_se3": pa.list_(pa.float64(), len(PoseSE3Index)),
    f"{FCAM_NAME(name)}.timestamp_us": pa.int64(),
}

FISHEYE_CAMERA_DATA_COLUMN: Callable[[str], str] = lambda name: f"{FCAM_NAME(name)}.data"
FISHEYE_CAMERA_EXTRINSIC_COLUMN: Callable[[str], str] = lambda name: f"{FCAM_NAME(name)}.state_se3"
FISHEYE_CAMERA_TIMESTAMP_COLUMN: Callable[[str], str] = lambda name: f"{FCAM_NAME(name)}.timestamp_us"

# Lidar
# ----------------------------------------------------------------------------------------------------------------------
LIDAR_NAME: Callable[[str], str] = lambda name: f"lidar.{name}"

LIDAR_STRING_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{LIDAR_NAME(name)}.data": pa.string(),
    f"{LIDAR_NAME(name)}.start_timestamp_us": pa.int64(),
    f"{LIDAR_NAME(name)}.end_timestamp_us": pa.int64(),
}

LIDAR_BINARY_SCHEMA_DICT: Callable[[str], dict] = lambda name: {
    f"{LIDAR_NAME(name)}.point_cloud_3d": pa.binary(),
    f"{LIDAR_NAME(name)}.point_cloud_features": pa.binary(),
    f"{LIDAR_NAME(name)}.start_timestamp_us": pa.int64(),
    f"{LIDAR_NAME(name)}.end_timestamp_us": pa.int64(),
}

LIDAR_PATH_COLUMN: Callable[[str], str] = lambda name: f"{LIDAR_NAME(name)}.data"
LIDAR_POINT_CLOUD_COLUMN: Callable[[str], str] = lambda name: f"{LIDAR_NAME(name)}.point_cloud_3d"
LIDAR_POINT_CLOUD_FEATURE_COLUMN: Callable[[str], str] = lambda name: f"{LIDAR_NAME(name)}.point_cloud_features"

# Auxiliary data
# ----------------------------------------------------------------------------------------------------------------------
AUX_NAME: Final[str] = "aux"
AUX_SCHEMA_DICT: Final[dict] = {
    f"{AUX_NAME}.data": pa.binary(),
}

ROUTE_LANE_GROUP_IDS_COLUMN: Final[str] = f"{AUX_NAME}.route_lane_group_ids"
SCENARIO_TAGS_COLUMN: Final[str] = f"{AUX_NAME}.scenario_tags"
