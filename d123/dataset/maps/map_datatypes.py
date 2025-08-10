from __future__ import annotations

from d123.common.utils.enums import SerialIntEnum

# TODO: Add stop pads or stop lines.


class MapLayer(SerialIntEnum):
    """
    Enum for AbstractMapSurface.
    """

    LANE = 0
    LANE_GROUP = 1
    INTERSECTION = 2
    CROSSWALK = 3
    WALKWAY = 4
    CARPARK = 5
    GENERIC_DRIVABLE = 6
    STOP_LINE = 7
    ROAD_EDGE = 8
    ROAD_LINE = 9


class LaneType(SerialIntEnum):
    """
    Enum for LaneType.
    NOTE: We use the lane types from Waymo.
    https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/protos/map.proto#L147
    """

    UNDEFINED = 0
    FREEWAY = 1
    SURFACE_STREET = 2
    BIKE_LANE = 3


class RoadEdgeType(SerialIntEnum):
    """
    Enum for RoadEdgeType.
    NOTE: We use the road line types from Waymo.
    https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L188
    """

    UNKNOWN = 0
    ROAD_EDGE_BOUNDARY = 1
    ROAD_EDGE_MEDIAN = 2


class RoadLineType(SerialIntEnum):
    """
    Enum for RoadLineType.
    NOTE: We use the road line types from Waymo.
    https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L208
    """

    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8
