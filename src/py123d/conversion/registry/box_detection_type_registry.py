from py123d.datatypes.detections.box_detection_types import AbstractBoxDetectionType

BOX_DETECTION_TYPE_REGISTRY = {}


def register_box_detection_type(enum_class):
    BOX_DETECTION_TYPE_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


@register_box_detection_type
class AV2SensorBoxDetectionType(AbstractBoxDetectionType):
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


@register_box_detection_type
class KITTI360BoxDetectionType(AbstractBoxDetectionType):
    pass


@register_box_detection_type
class NuPlanBoxDetectionType(AbstractBoxDetectionType):

    VEHICLE = 0
    BICYCLE = 1
    PEDESTRIAN = 2
    TRAFFIC_CONE = 3
    BARRIER = 4
    CZONE_SIGN = 5
    GENERIC_OBJECT = 6


@register_box_detection_type
class NuScenesBoxDetectionType(AbstractBoxDetectionType):
    pass


class PandasetBoxDetectionType(AbstractBoxDetectionType):

    ANIMALS_BIRD = 0
    ANIMALS_OTHER = 1
    BICYCLE = 2
    BUS = 3
    CAR = 4
    CONES = 5
    CONSTRUCTION_SIGNS = 6
    EMERGENCY_VEHICLE = 7
    MEDIUM_SIZED_TRUCK = 8
    MOTORCYCLE = 9
    MOTORIZED_SCOOTER = 10
    OTHER_VEHICLE_CONSTRUCTION_VEHICLE = 11
    OTHER_VEHICLE_PEDICAB = 12
    OTHER_VEHICLE_UNCOMMON = 13
    PEDESTRIAN = 14
    PEDESTRIAN_WITH_OBJECT = 15
    PERSONAL_MOBILITY_DEVICE = 16
    PICKUP_TRUCK = 17
    PYLONS = 18
    ROAD_BARRIERS = 19
    ROLLING_CONTAINERS = 20
    SEMI_TRUCK = 21
    SIGNS = 22
    TEMPORARY_CONSTRUCTION_BARRIERS = 23
    TOWED_OBJECT = 24
    TRAIN = 25
    TRAM_SUBWAY = 26


class WOPDBoxDetectionType(AbstractBoxDetectionType):
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto#L63-L69

    TYPE_UNKNOWN = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_SIGN = 3
    TYPE_CYCLIST = 4
