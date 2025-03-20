from __future__ import annotations

from enum import IntEnum


class MapObjectType(IntEnum):
    """
    Enum for SemanticMapLayers.
    """

    LANE = 0
    LANE_GROUP = 1
    INTERSECTION = 2
    CROSSWALK = 3
    WALKWAYS = 4
    PARKING = 5
    GENERIC_DRIVABLE_AREA = 6

    # TODO


class TrafficLightStatusType(IntEnum):
    """
    Enum for TrafficLightStatusType.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3

    def serialize(self) -> str:
        """Serialize the type when saving."""
        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """Deserialize the type when loading from a string."""
        return TrafficLightStatusType.__members__[key]


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
