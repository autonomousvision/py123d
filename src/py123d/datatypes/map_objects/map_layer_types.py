from __future__ import annotations

from py123d.common.utils.enums import SerialIntEnum

# TODO: Add stop pads or stop lines.
# - Add type for stop zones.
# - Add type for carparks, e.g. outline, driveway (Waymo), or other types.
# - Check if intersections should have types.
# - Use consistent naming conventions unknown, undefined, none, etc.


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
    STOP_ZONE = 7
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
    TODO: Use the Argoverse 2 road line types.
    https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L208
    """

    NONE = 0
    UNKNOWN = 1
    DASH_SOLID_YELLOW = 2
    DASH_SOLID_WHITE = 3
    DASHED_WHITE = 4
    DASHED_YELLOW = 5
    DOUBLE_SOLID_YELLOW = 6
    DOUBLE_SOLID_WHITE = 7
    DOUBLE_DASH_YELLOW = 8
    DOUBLE_DASH_WHITE = 9
    SOLID_YELLOW = 10
    SOLID_WHITE = 11
    SOLID_DASH_WHITE = 12
    SOLID_DASH_YELLOW = 13
    SOLID_BLUE = 14
