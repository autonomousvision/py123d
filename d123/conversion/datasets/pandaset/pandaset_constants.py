from typing import Dict, List

from d123.common.utils.enums import SerialIntEnum
from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType

PANDASET_SPLITS: List[str] = ["pandaset_train", "pandaset_val", "pandaset_test"]

PANDASET_CAMERA_MAPPING: Dict[str, PinholeCameraType] = {
    "front_camera": PinholeCameraType.CAM_F0,
    "back_camera": PinholeCameraType.CAM_B0,
    "front_left_camera": PinholeCameraType.CAM_L0,
    "front_right_camera": PinholeCameraType.CAM_R0,
    "left_camera": PinholeCameraType.CAM_L1,
    "right_camera": PinholeCameraType.CAM_R1,
}


class PandasetBoxDetectionType(SerialIntEnum):

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


PANDASET_BOX_DETECTION_FROM_STR: Dict[str, PandasetBoxDetectionType] = {
    "Animals - Bird": PandasetBoxDetectionType.ANIMALS_BIRD,
    "Animals - Other": PandasetBoxDetectionType.ANIMALS_OTHER,
    "Bicycle": PandasetBoxDetectionType.BICYCLE,
    "Bus": PandasetBoxDetectionType.BUS,
    "Car": PandasetBoxDetectionType.CAR,
    "Cones": PandasetBoxDetectionType.CONES,
    "Construction Signs": PandasetBoxDetectionType.CONSTRUCTION_SIGNS,
    "Emergency Vehicle": PandasetBoxDetectionType.EMERGENCY_VEHICLE,
    "Medium-sized Truck": PandasetBoxDetectionType.MEDIUM_SIZED_TRUCK,
    "Motorcycle": PandasetBoxDetectionType.MOTORCYCLE,
    "Motorized Scooter": PandasetBoxDetectionType.MOTORIZED_SCOOTER,
    "Other Vehicle - Construction Vehicle": PandasetBoxDetectionType.OTHER_VEHICLE_CONSTRUCTION_VEHICLE,
    "Other Vehicle - Pedicab": PandasetBoxDetectionType.OTHER_VEHICLE_PEDICAB,
    "Other Vehicle - Uncommon": PandasetBoxDetectionType.OTHER_VEHICLE_UNCOMMON,
    "Pedestrian": PandasetBoxDetectionType.PEDESTRIAN,
    "Pedestrian with Object": PandasetBoxDetectionType.PEDESTRIAN_WITH_OBJECT,
    "Personal Mobility Device": PandasetBoxDetectionType.PERSONAL_MOBILITY_DEVICE,
    "Pickup Truck": PandasetBoxDetectionType.PICKUP_TRUCK,
    "Pylons": PandasetBoxDetectionType.PYLONS,
    "Road Barriers": PandasetBoxDetectionType.ROAD_BARRIERS,
    "Rolling Containers": PandasetBoxDetectionType.ROLLING_CONTAINERS,
    "Semi-truck": PandasetBoxDetectionType.SEMI_TRUCK,
    "Signs": PandasetBoxDetectionType.SIGNS,
    "Temporary Construction Barriers": PandasetBoxDetectionType.TEMPORARY_CONSTRUCTION_BARRIERS,
    "Towed Object": PandasetBoxDetectionType.TOWED_OBJECT,
    "Train": PandasetBoxDetectionType.TRAIN,
    "Tram / Subway": PandasetBoxDetectionType.TRAM_SUBWAY,
}


PANDASET_BOX_DETECTION_TO_DEFAULT: Dict[PandasetBoxDetectionType, DetectionType] = {
    PandasetBoxDetectionType.ANIMALS_BIRD: DetectionType.GENERIC_OBJECT,  # TODO: Adjust default types
    PandasetBoxDetectionType.ANIMALS_OTHER: DetectionType.GENERIC_OBJECT,  # TODO: Adjust default types
    PandasetBoxDetectionType.BICYCLE: DetectionType.BICYCLE,
    PandasetBoxDetectionType.BUS: DetectionType.VEHICLE,
    PandasetBoxDetectionType.CAR: DetectionType.VEHICLE,
    PandasetBoxDetectionType.CONES: DetectionType.TRAFFIC_CONE,
    PandasetBoxDetectionType.CONSTRUCTION_SIGNS: DetectionType.CZONE_SIGN,
    PandasetBoxDetectionType.EMERGENCY_VEHICLE: DetectionType.VEHICLE,
    PandasetBoxDetectionType.MEDIUM_SIZED_TRUCK: DetectionType.VEHICLE,
    PandasetBoxDetectionType.MOTORCYCLE: DetectionType.BICYCLE,
    PandasetBoxDetectionType.MOTORIZED_SCOOTER: DetectionType.BICYCLE,
    PandasetBoxDetectionType.OTHER_VEHICLE_CONSTRUCTION_VEHICLE: DetectionType.VEHICLE,
    PandasetBoxDetectionType.OTHER_VEHICLE_PEDICAB: DetectionType.BICYCLE,
    PandasetBoxDetectionType.OTHER_VEHICLE_UNCOMMON: DetectionType.VEHICLE,
    PandasetBoxDetectionType.PEDESTRIAN: DetectionType.PEDESTRIAN,
    PandasetBoxDetectionType.PEDESTRIAN_WITH_OBJECT: DetectionType.PEDESTRIAN,
    PandasetBoxDetectionType.PERSONAL_MOBILITY_DEVICE: DetectionType.BICYCLE,
    PandasetBoxDetectionType.PICKUP_TRUCK: DetectionType.VEHICLE,
    PandasetBoxDetectionType.PYLONS: DetectionType.TRAFFIC_CONE,
    PandasetBoxDetectionType.ROAD_BARRIERS: DetectionType.BARRIER,
    PandasetBoxDetectionType.ROLLING_CONTAINERS: DetectionType.GENERIC_OBJECT,
    PandasetBoxDetectionType.SEMI_TRUCK: DetectionType.VEHICLE,
    PandasetBoxDetectionType.SIGNS: DetectionType.SIGN,
    PandasetBoxDetectionType.TEMPORARY_CONSTRUCTION_BARRIERS: DetectionType.BARRIER,
    PandasetBoxDetectionType.TOWED_OBJECT: DetectionType.VEHICLE,
    PandasetBoxDetectionType.TRAIN: DetectionType.GENERIC_OBJECT,  # TODO: Adjust default types
    PandasetBoxDetectionType.TRAM_SUBWAY: DetectionType.GENERIC_OBJECT,  # TODO: Adjust default types
}


PANDASET_LOG_NAMES: List[str] = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "008",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
    "021",
    "023",
    "024",
    "027",
    "028",
    "029",
    "030",
    "032",
    "033",
    "034",
    "035",
    "037",
    "038",
    "039",
    "040",
    "041",
    "042",
    "043",
    "044",
    "045",
    "046",
    "047",
    "048",
    "050",
    "051",
    "052",
    "053",
    "054",
    "055",
    "056",
    "057",
    "058",
    "059",
    "062",
    "063",
    "064",
    "065",
    "066",
    "067",
    "068",
    "069",
    "070",
    "071",
    "072",
    "073",
    "074",
    "077",
    "078",
    "079",
    "080",
    "084",
    "085",
    "086",
    "088",
    "089",
    "090",
    "091",
    "092",
    "093",
    "094",
    "095",
    "097",
    "098",
    "099",
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "109",
    "110",
    "112",
    "113",
    "115",
    "116",
    "117",
    "119",
    "120",
    "122",
    "123",
    "124",
    "139",
    "149",
    "158",
]
