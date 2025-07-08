from __future__ import annotations

from asim.common.utils.enums import SerialIntEnum

# TODO: Add stop pads or stop lines.


class MapSurfaceType(SerialIntEnum):
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
