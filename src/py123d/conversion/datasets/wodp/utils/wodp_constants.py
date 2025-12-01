from typing import Dict, List

from py123d.conversion.registry.box_detection_label_registry import WODPBoxDetectionLabel
from py123d.datatypes.map_objects.map_layer_types import LaneType, RoadEdgeType, RoadLineType
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType

WODP_AVAILABLE_SPLITS: List[str] = ["wodp_train", "wodp_val", "wodp_test"]

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto#L63
WODP_DETECTION_NAME_DICT: Dict[int, WODPBoxDetectionLabel] = {
    0: WODPBoxDetectionLabel.TYPE_UNKNOWN,
    1: WODPBoxDetectionLabel.TYPE_VEHICLE,
    2: WODPBoxDetectionLabel.TYPE_PEDESTRIAN,
    3: WODPBoxDetectionLabel.TYPE_SIGN,
    4: WODPBoxDetectionLabel.TYPE_CYCLIST,
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L50
WODP_CAMERA_TYPES: Dict[int, PinholeCameraType] = {
    1: PinholeCameraType.PCAM_F0,  # front_camera
    2: PinholeCameraType.PCAM_L0,  # front_left_camera
    3: PinholeCameraType.PCAM_R0,  # front_right_camera
    4: PinholeCameraType.PCAM_L1,  # left_camera
    5: PinholeCameraType.PCAM_R1,  # right_camera
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L66
WODP_LIDAR_TYPES: Dict[int, LiDARType] = {
    0: LiDARType.LIDAR_UNKNOWN,  # UNKNOWN
    1: LiDARType.LIDAR_TOP,  # TOP
    2: LiDARType.LIDAR_FRONT,  # FRONT
    3: LiDARType.LIDAR_SIDE_LEFT,  # SIDE_LEFT
    4: LiDARType.LIDAR_SIDE_RIGHT,  # SIDE_RIGHT
    5: LiDARType.LIDAR_BACK,  # REAR
}

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
