from typing import Dict, Final, Set

from d123.common.utils.enums import SerialIntEnum
from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.maps.map_datatypes import RoadLineType
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType

AV2_SENSOR_SPLITS: Set[str] = {
    "av2-sensor_train",
    "av2-sensor_val",
    "av2-sensor_test",
    "av2-sensor-mini_train",
    "av2-sensor-mini_val",
    "av2-sensor-mini_test",
}


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
    AV2SensorBoxDetectionType.ANIMAL: DetectionType.GENERIC_OBJECT,
    AV2SensorBoxDetectionType.ARTICULATED_BUS: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.BICYCLE: DetectionType.BICYCLE,
    AV2SensorBoxDetectionType.BICYCLIST: DetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.BOLLARD: DetectionType.BARRIER,
    AV2SensorBoxDetectionType.BOX_TRUCK: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.BUS: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.CONSTRUCTION_BARREL: DetectionType.BARRIER,
    AV2SensorBoxDetectionType.CONSTRUCTION_CONE: DetectionType.TRAFFIC_CONE,
    AV2SensorBoxDetectionType.DOG: DetectionType.GENERIC_OBJECT,
    AV2SensorBoxDetectionType.LARGE_VEHICLE: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.MESSAGE_BOARD_TRAILER: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.MOBILE_PEDESTRIAN_CROSSING_SIGN: DetectionType.CZONE_SIGN,
    AV2SensorBoxDetectionType.MOTORCYCLE: DetectionType.BICYCLE,
    AV2SensorBoxDetectionType.MOTORCYCLIST: DetectionType.BICYCLE,
    AV2SensorBoxDetectionType.OFFICIAL_SIGNALER: DetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.PEDESTRIAN: DetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.RAILED_VEHICLE: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.REGULAR_VEHICLE: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.SCHOOL_BUS: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.SIGN: DetectionType.SIGN,
    AV2SensorBoxDetectionType.STOP_SIGN: DetectionType.SIGN,
    AV2SensorBoxDetectionType.STROLLER: DetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.TRAFFIC_LIGHT_TRAILER: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.TRUCK: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.TRUCK_CAB: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.VEHICULAR_TRAILER: DetectionType.VEHICLE,
    AV2SensorBoxDetectionType.WHEELCHAIR: DetectionType.PEDESTRIAN,
    AV2SensorBoxDetectionType.WHEELED_DEVICE: DetectionType.GENERIC_OBJECT,
    AV2SensorBoxDetectionType.WHEELED_RIDER: DetectionType.BICYCLE,
}


AV2_CAMERA_TYPE_MAPPING: Dict[str, PinholeCameraType] = {
    "ring_front_center": PinholeCameraType.CAM_F0,
    "ring_front_left": PinholeCameraType.CAM_L0,
    "ring_front_right": PinholeCameraType.CAM_R0,
    "ring_side_left": PinholeCameraType.CAM_L1,
    "ring_side_right": PinholeCameraType.CAM_R1,
    "ring_rear_left": PinholeCameraType.CAM_L2,
    "ring_rear_right": PinholeCameraType.CAM_R2,
    "stereo_front_left": PinholeCameraType.CAM_STEREO_L,
    "stereo_front_right": PinholeCameraType.CAM_STEREO_R,
}


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
