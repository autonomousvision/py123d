from typing import Dict, Final, Set

from py123d.datatypes.map.map_datatypes import RoadLineType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType

AV2_SENSOR_SPLITS: Set[str] = {"av2-sensor_train", "av2-sensor_val", "av2-sensor_test"}


AV2_CAMERA_TYPE_MAPPING: Dict[str, PinholeCameraType] = {
    "ring_front_center": PinholeCameraType.PCAM_F0,
    "ring_front_left": PinholeCameraType.PCAM_L0,
    "ring_front_right": PinholeCameraType.PCAM_R0,
    "ring_side_left": PinholeCameraType.PCAM_L1,
    "ring_side_right": PinholeCameraType.PCAM_R1,
    "ring_rear_left": PinholeCameraType.PCAM_L2,
    "ring_rear_right": PinholeCameraType.PCAM_R2,
    "stereo_front_left": PinholeCameraType.PCAM_STEREO_L,
    "stereo_front_right": PinholeCameraType.PCAM_STEREO_R,
}

# AV2_LIDAR_TYPES: Dict[str, str] = {


AV2_ROAD_LINE_TYPE_MAPPING: Dict[str, RoadLineType] = {
    "NONE": RoadLineType.NONE,
    "UNKNOWN": RoadLineType.UNKNOWN,
    "DASH_SOLID_YELLOW": RoadLineType.DASH_SOLID_YELLOW,
    "DASH_SOLID_WHITE": RoadLineType.DASH_SOLID_WHITE,
    "DASHED_WHITE": RoadLineType.DASHED_WHITE,
    "DASHED_YELLOW": RoadLineType.DASHED_YELLOW,
    "DOUBLE_SOLID_YELLOW": RoadLineType.DOUBLE_SOLID_YELLOW,
    "DOUBLE_SOLID_WHITE": RoadLineType.DOUBLE_SOLID_WHITE,
    "DOUBLE_DASH_YELLOW": RoadLineType.DOUBLE_DASH_YELLOW,
    "DOUBLE_DASH_WHITE": RoadLineType.DOUBLE_DASH_WHITE,
    "SOLID_YELLOW": RoadLineType.SOLID_YELLOW,
    "SOLID_WHITE": RoadLineType.SOLID_WHITE,
    "SOLID_DASH_WHITE": RoadLineType.SOLID_DASH_WHITE,
    "SOLID_DASH_YELLOW": RoadLineType.SOLID_DASH_YELLOW,
    "SOLID_BLUE": RoadLineType.SOLID_BLUE,
}


AV2_SENSOR_CAM_SHUTTER_INTERVAL_MS: Final[float] = 50.0
AV2_SENSOR_LIDAR_SWEEP_INTERVAL_W_BUFFER_NS: Final[float] = 102000000.0
