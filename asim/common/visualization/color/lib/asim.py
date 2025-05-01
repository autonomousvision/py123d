from typing import Dict

from asim.common.visualization.color.color import Color
from asim.dataset.maps.map_datatypes import MapSurfaceType

LIGHT_GREY = Color("#D3D3D3")

MAP_SURFACE_COLORS: Dict[MapSurfaceType, Color] = {
    MapSurfaceType.LANE: LIGHT_GREY,
    MapSurfaceType.LANE_GROUP: LIGHT_GREY,
    MapSurfaceType.INTERSECTION: LIGHT_GREY,
    MapSurfaceType.CROSSWALK: Color("#b07aa1"),
    MapSurfaceType.WALKWAY: Color("#d4d19e"),
    MapSurfaceType.CARPARK: Color("#b9d3b4"),
    MapSurfaceType.GENERIC_DRIVABLE: LIGHT_GREY,
}
