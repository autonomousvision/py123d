from __future__ import annotations


from asim.common.utils.enums import SerialIntEnum


class MapObjectType(SerialIntEnum):
    """
    Enum for SemanticMapLayers.
    """

    LANE = 0
    LANE_GROUP = 1
    INTERSECTION = 2
    CROSSWALK = 3
    WALKWAYS = 4
    CARPARK = 5
    GENERIC_DRIVABLE = 6


class TrafficLightStatusType(SerialIntEnum):
    """
    Enum for TrafficLightStatusType.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3


# STOP_LINE = 2
# TURN_STOP = 3
# CROSSWALK = 4
# DRIVABLE_AREA = 5
# YIELD = 6
# TRAFFIC_LIGHT = 7
# STOP_SIGN = 8
# EXTENDED_PUDO = 9
# SPEED_BUMP = 10
# LANE_CONNECTOR = 11
# BASELINE_PATHS = 12
# BOUNDARIES = 13
# WALKWAYS = 14
# CARPARK_AREA = 15
# PUDO = 16
# ROADBLOCK = 17
# ROADBLOCK_CONNECTOR = 18
