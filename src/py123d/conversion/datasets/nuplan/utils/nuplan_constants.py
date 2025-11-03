from typing import Dict, Final, List, Set

from py123d.datatypes.detections.box_detection_types import BoxDetectionType
from py123d.datatypes.detections.traffic_light_detections import TrafficLightStatus
from py123d.datatypes.maps.map_datatypes import RoadLineType
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.datatypes.time.time_point import TimePoint

NUPLAN_DEFAULT_DT: Final[float] = 0.05

NUPLAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "red": TrafficLightStatus.RED,
    "unknown": TrafficLightStatus.UNKNOWN,
}


NUPLAN_DETECTION_NAME_DICT = {
    "vehicle": BoxDetectionType.VEHICLE,
    "bicycle": BoxDetectionType.BICYCLE,
    "pedestrian": BoxDetectionType.PEDESTRIAN,
    "traffic_cone": BoxDetectionType.TRAFFIC_CONE,
    "barrier": BoxDetectionType.BARRIER,
    "czone_sign": BoxDetectionType.CZONE_SIGN,
    "generic_object": BoxDetectionType.GENERIC_OBJECT,
}

# https://github.com/motional/nuplan-devkit/blob/e9241677997dd86bfc0bcd44817ab04fe631405b/nuplan/database/nuplan_db_orm/utils.py#L1129-L1135
NUPLAN_LIDAR_DICT = {
    0: LiDARType.LIDAR_TOP,
    1: LiDARType.LIDAR_SIDE_RIGHT,
    2: LiDARType.LIDAR_SIDE_LEFT,
    3: LiDARType.LIDAR_BACK,
    4: LiDARType.LIDAR_FRONT,
}

NUPLAN_DATA_SPLITS: Set[str] = {
    "nuplan_train",
    "nuplan_val",
    "nuplan_test",
    "nuplan-mini_train",
    "nuplan-mini_val",
    "nuplan-mini_test",
    "nuplan-private_test",  # TODO: remove, not publicly available
}

NUPLAN_MAP_LOCATIONS: List[str] = [
    "sg-one-north",
    "us-ma-boston",
    "us-nv-las-vegas-strip",
    "us-pa-pittsburgh-hazelwood",
]

NUPLAN_MAP_LOCATION_FILES: Dict[str, str] = {
    "sg-one-north": "sg-one-north/9.17.1964/map.gpkg",
    "us-ma-boston": "us-ma-boston/9.12.1817/map.gpkg",
    "us-nv-las-vegas-strip": "us-nv-las-vegas-strip/9.15.1915/map.gpkg",
    "us-pa-pittsburgh-hazelwood": "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg",
}


NUPLAN_MAP_GPKG_LAYERS: Set[str] = {
    "baseline_paths",
    "carpark_areas",
    "generic_drivable_areas",
    "dubins_nodes",
    "lane_connectors",
    "intersections",
    "boundaries",
    "crosswalks",
    "lanes_polygons",
    "lane_group_connectors",
    "lane_groups_polygons",
    "road_segments",
    "stop_polygons",
    "traffic_lights",
    "walkways",
    "gen_lane_connectors_scaled_width_polygons",
}

NUPLAN_ROAD_LINE_CONVERSION = {
    0: RoadLineType.DASHED_WHITE,
    2: RoadLineType.SOLID_WHITE,
    3: RoadLineType.UNKNOWN,
}


NUPLAN_ROLLING_SHUTTER_S: Final[TimePoint] = TimePoint.from_s(1 / 60)
