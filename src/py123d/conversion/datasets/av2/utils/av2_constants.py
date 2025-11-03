from typing import Dict, Final, Set

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.detections.box_detection_types import BoxDetectionType
from py123d.datatypes.maps.map_datatypes import RoadLineType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType

AV2_SENSOR_SPLITS: Set[str] = {"av2-sensor_train", "av2-sensor_val", "av2-sensor_test"}


class AV2SensorBoxDetectionType(SerialIntEnum):
    """Sensor dataset annotation categories."""

    ANIMAL = 1
    ARTICULATED_BUS = 2
    BICYCLE = 3
    BICYCLIST = 4
    BOLLARD = 5
    BOX_TRUCK = 6
    BUS = 7
    CONSTRUCTION_BARREL = 8
    CONSTRUCTION_CONE = 9
    DOG = 10
    LARGE_VEHICLE = 11
    MESSAGE_BOARD_TRAILER = 12
    MOBILE_PEDESTRIAN_CROSSING_SIGN = 13
    MOTORCYCLE = 14
    MOTORCYCLIST = 15
    OFFICIAL_SIGNALER = 16
    PEDESTRIAN = 17
    RAILED_VEHICLE = 18
    REGULAR_VEHICLE = 19
    SCHOOL_BUS = 20
    SIGN = 21
    STOP_SIGN = 22
    STROLLER = 23
    TRAFFIC_LIGHT_TRAILER = 24
    TRUCK = 25
    TRUCK_CAB = 26
    VEHICULAR_TRAILER = 27
    WHEELCHAIR = 28
    WHEELED_DEVICE = 29
    WHEELED_RIDER = 30


# Mapping from AV2SensorBoxDetectionType to general DetectionType
# TODO: Change the detection types. Multiple mistakes, e.g. animals/dogs are not generic objects.
AV2_TO_DETECTION_TYPE = {
    AV2SensorBoxDetectionType.ANIMAL: BoxDetectionType.GENERIC_OBJECT,
    AV2SensorBoxDetectionType.ARTICULATED_BUS: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.BICYCLE: BoxDetectionType.BICYCLE,
    AV2SensorBoxDetectionType.BICYCLIST: BoxDetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.BOLLARD: BoxDetectionType.BARRIER,
    AV2SensorBoxDetectionType.BOX_TRUCK: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.BUS: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.CONSTRUCTION_BARREL: BoxDetectionType.BARRIER,
    AV2SensorBoxDetectionType.CONSTRUCTION_CONE: BoxDetectionType.TRAFFIC_CONE,
    AV2SensorBoxDetectionType.DOG: BoxDetectionType.GENERIC_OBJECT,
    AV2SensorBoxDetectionType.LARGE_VEHICLE: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.MESSAGE_BOARD_TRAILER: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.MOBILE_PEDESTRIAN_CROSSING_SIGN: BoxDetectionType.CZONE_SIGN,
    AV2SensorBoxDetectionType.MOTORCYCLE: BoxDetectionType.BICYCLE,
    AV2SensorBoxDetectionType.MOTORCYCLIST: BoxDetectionType.BICYCLE,
    AV2SensorBoxDetectionType.OFFICIAL_SIGNALER: BoxDetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.PEDESTRIAN: BoxDetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.RAILED_VEHICLE: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.REGULAR_VEHICLE: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.SCHOOL_BUS: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.SIGN: BoxDetectionType.SIGN,
    AV2SensorBoxDetectionType.STOP_SIGN: BoxDetectionType.SIGN,
    AV2SensorBoxDetectionType.STROLLER: BoxDetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.TRAFFIC_LIGHT_TRAILER: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.TRUCK: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.TRUCK_CAB: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.VEHICULAR_TRAILER: BoxDetectionType.VEHICLE,
    AV2SensorBoxDetectionType.WHEELCHAIR: BoxDetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.WHEELED_DEVICE: BoxDetectionType.GENERIC_OBJECT,
    AV2SensorBoxDetectionType.WHEELED_RIDER: BoxDetectionType.BICYCLE,
}


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
