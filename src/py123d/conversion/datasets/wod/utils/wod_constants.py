from typing import Dict, List

from py123d.datatypes.detections import TrafficLightStatus
from py123d.datatypes.map_objects import LaneType, RoadEdgeType, RoadLineType
from py123d.datatypes.sensors import LiDARType, PinholeCameraType

# Map features:
# ----------------------------------------------------------------------------------------------------------------------

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L206
WAYMO_ROAD_LINE_TYPE_CONVERSION: Dict[int, RoadLineType] = {
    0: RoadLineType.UNKNOWN,  # UNKNOWN
    1: RoadLineType.DASHED_WHITE,  # BROKEN_SINGLE_WHITE
    2: RoadLineType.SOLID_WHITE,  # SOLID_SINGLE_WHITE
    3: RoadLineType.DOUBLE_SOLID_WHITE,  # SOLID_DOUBLE_WHITE
    4: RoadLineType.DASHED_YELLOW,  # BROKEN_SINGLE_YELLOW
    5: RoadLineType.DOUBLE_DASH_YELLOW,  # BROKEN_DOUBLE_YELLOW
    6: RoadLineType.SOLID_YELLOW,  # SOLID_SINGLE_YELLOW
    7: RoadLineType.DOUBLE_SOLID_YELLOW,  # SOLID_DOUBLE_YELLOW
    8: RoadLineType.DOUBLE_DASH_YELLOW,  # PASSING_DOUBLE_YELLOW
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L186
WAYMO_ROAD_EDGE_TYPE_CONVERSION: Dict[int, RoadEdgeType] = {
    0: RoadEdgeType.UNKNOWN,
    1: RoadEdgeType.ROAD_EDGE_BOUNDARY,
    2: RoadEdgeType.ROAD_EDGE_MEDIAN,
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L147
WAYMO_LANE_TYPE_CONVERSION: Dict[int, LaneType] = {
    0: LaneType.UNDEFINED,
    1: LaneType.FREEWAY,
    2: LaneType.SURFACE_STREET,
    3: LaneType.BIKE_LANE,
}


# Perception:
# ----------------------------------------------------------------------------------------------------------------------
WOD_PERCEPTION_AVAILABLE_SPLITS: List[str] = ["wod-perception_train", "wod-perception_val", "wod-perception_test"]

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L50
WOD_PERCEPTION_CAMERA_TYPES: Dict[int, PinholeCameraType] = {
    1: PinholeCameraType.PCAM_F0,  # front_camera
    2: PinholeCameraType.PCAM_L0,  # front_left_camera
    3: PinholeCameraType.PCAM_R0,  # front_right_camera
    4: PinholeCameraType.PCAM_L1,  # left_camera
    5: PinholeCameraType.PCAM_R1,  # right_camera
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L66
WOD_PERCEPTION_LIDAR_TYPES: Dict[int, LiDARType] = {
    0: LiDARType.LIDAR_UNKNOWN,  # UNKNOWN
    1: LiDARType.LIDAR_TOP,  # TOP
    2: LiDARType.LIDAR_FRONT,  # FRONT
    3: LiDARType.LIDAR_SIDE_LEFT,  # SIDE_LEFT
    4: LiDARType.LIDAR_SIDE_RIGHT,  # SIDE_RIGHT
    5: LiDARType.LIDAR_BACK,  # REAR
}


# Motion:
# ----------------------------------------------------------------------------------------------------------------------

WOD_MOTION_AVAILABLE_SPLITS: List[str] = ["wod-motion_train", "wod-motion_val", "wod-motion_test"]

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L39
WOD_MOTION_TRAFFIC_LIGHT_MAPPING: Dict[int, TrafficLightStatus] = {
    0: TrafficLightStatus.UNKNOWN,  # LANE_STATE_UNKNOWN
    1: TrafficLightStatus.RED,  # LANE_STATE_ARROW_STOP
    2: TrafficLightStatus.YELLOW,  # LANE_STATE_ARROW_CAUTION
    3: TrafficLightStatus.GREEN,  # LANE_STATE_ARROW_GO
    4: TrafficLightStatus.RED,  # LANE_STATE_STOP
    5: TrafficLightStatus.YELLOW,  # LANE_STATE_CAUTION
    6: TrafficLightStatus.GREEN,  # LANE_STATE_GO
    7: TrafficLightStatus.RED,  # LANE_STATE_FLASHING_STOP
    8: TrafficLightStatus.YELLOW,  # LANE_STATE_FLASHING_CAUTION
}
