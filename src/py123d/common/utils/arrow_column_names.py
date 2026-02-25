from typing import Callable, Final, List

# Essential Columns
# ----------------------------------------------------------------------------------------------------------------------
UUID_COLUMN: Final[str] = "uuid"
TIMESTAMP_US_COLUMN: Final[str] = "timestamp_us"

# Ego State SE3
# ----------------------------------------------------------------------------------------------------------------------
EGO_IMU_SE3_COLUMN: Final[str] = "ego.imu_se3"
EGO_DYNAMIC_STATE_SE3_COLUMN: Final[str] = "ego.dynamic_state_se3"

EGO_STATE_SE3_COLUMNS: Final[List[str]] = [EGO_IMU_SE3_COLUMN, EGO_DYNAMIC_STATE_SE3_COLUMN]


# Box Detections SE3
# ----------------------------------------------------------------------------------------------------------------------
BOX_DETECTIONS_PREFIX: Final[str] = "box_detections"
BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN: Final[str] = f"{BOX_DETECTIONS_PREFIX}.bounding_box_se3"
BOX_DETECTIONS_TOKEN_COLUMN: Final[str] = f"{BOX_DETECTIONS_PREFIX}.token"
BOX_DETECTIONS_VELOCITY_3D_COLUMN: Final[str] = f"{BOX_DETECTIONS_PREFIX}.velocity_3d"
BOX_DETECTIONS_LABEL_COLUMN: Final[str] = f"{BOX_DETECTIONS_PREFIX}.label"
BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN: Final[str] = f"{BOX_DETECTIONS_PREFIX}.num_lidar_points"

BOX_DETECTIONS_SE3_COLUMNS: Final[List[str]] = [
    BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN,
    BOX_DETECTIONS_TOKEN_COLUMN,
    BOX_DETECTIONS_VELOCITY_3D_COLUMN,
    BOX_DETECTIONS_LABEL_COLUMN,
    BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN,
]

# Traffic Lights
# ----------------------------------------------------------------------------------------------------------------------
TRAFFIC_LIGHTS_PREFIX: Final[str] = "traffic_lights"
TRAFFIC_LIGHTS_LANE_ID_COLUMN: Final[str] = f"{TRAFFIC_LIGHTS_PREFIX}.lane_id"
TRAFFIC_LIGHTS_STATUS_COLUMN: Final[str] = f"{TRAFFIC_LIGHTS_PREFIX}.status"

TRAFFIC_LIGHTS_COLUMNS: Final[List[str]] = [TRAFFIC_LIGHTS_LANE_ID_COLUMN, TRAFFIC_LIGHTS_STATUS_COLUMN]

# Pinhole Cameras
# ----------------------------------------------------------------------------------------------------------------------
PINHOLE_PREFIX: Final[str] = "pinhole"
PINHOLE_CAMERA_DATA_COLUMN: Callable[[str], str] = lambda name: f"{PINHOLE_PREFIX}.{name}.data"
PINHOLE_CAMERA_EXTRINSIC_COLUMN: Callable[[str], str] = lambda name: f"{PINHOLE_PREFIX}.{name}.state_se3"
PINHOLE_CAMERA_TIMESTAMP_COLUMN: Callable[[str], str] = lambda name: f"{PINHOLE_PREFIX}.{name}.timestamp_us"

PINHOLE_CAMERA_COLUMNS: Callable[[str], List[str]] = lambda name: [
    PINHOLE_CAMERA_DATA_COLUMN(name),
    PINHOLE_CAMERA_EXTRINSIC_COLUMN(name),
]


# Fisheye MEI Cameras
# ----------------------------------------------------------------------------------------------------------------------
FISHEYE_PREFIX: Final[str] = "fisheye_mei"
FISHEYE_CAMERA_DATA_COLUMN: Callable[[str], str] = lambda name: f"{FISHEYE_PREFIX}.{name}.data"
FISHEYE_CAMERA_EXTRINSIC_COLUMN: Callable[[str], str] = lambda name: f"{FISHEYE_PREFIX}.{name}.state_se3"
FISHEYE_CAMERA_TIMESTAMP_COLUMN: Callable[[str], str] = lambda name: f"{FISHEYE_PREFIX}.{name}.timestamp_us"

FISHEYE_CAMERA_COLUMNS: Callable[[str], List[str]] = lambda name: [
    FISHEYE_CAMERA_DATA_COLUMN(name),
    FISHEYE_CAMERA_EXTRINSIC_COLUMN(name),
]


# Lidar
# ----------------------------------------------------------------------------------------------------------------------
LIDAR_PATH_COLUMN: Callable[[str], str] = lambda name: f"lidar.{name}.path"
LIDAR_POINT_CLOUD_COLUMN: Callable[[str], str] = lambda name: f"lidar.{name}.point_cloud_3d"
LIDAR_POINT_CLOUD_FEATURE_COLUMN: Callable[[str], str] = lambda name: f"lidar.{name}.point_cloud_features"


# Miscellaneous (Scenario Tags / Route)
# ----------------------------------------------------------------------------------------------------------------------
SCENARIO_TAGS_COLUMN: str = "scenario_tags"
ROUTE_LANE_GROUP_IDS_COLUMN: str = "route_lane_group_ids"
