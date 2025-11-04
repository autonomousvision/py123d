from typing import Dict

from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
from py123d.datatypes.detections.traffic_light_detections import TrafficLightStatus
from py123d.datatypes.maps.map_datatypes import MapLayer
from py123d.visualization.color.color import (
    BLACK,
    DARKER_GREY,
    ELLIS_5,
    LIGHT_GREY,
    NEW_TAB_10,
    TAB_10,
    WHITE,
    Color,
)
from py123d.visualization.color.config import PlotConfig

HEADING_MARKER_STYLE: str = "^"  # "^": triangle, "-": line

MAP_SURFACE_CONFIG: Dict[MapLayer, PlotConfig] = {
    MapLayer.LANE: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=1.0,
        line_style="-",
        zorder=1,
    ),
    MapLayer.LANE_GROUP: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=1.0,
        line_style="-",
        zorder=1,
    ),
    MapLayer.INTERSECTION: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=1.0,
        line_style="-",
        zorder=1,
    ),
    MapLayer.CROSSWALK: PlotConfig(
        fill_color=Color("#c69fbb"),
        fill_color_alpha=1.0,
        line_color=Color("#c69fbb"),
        line_color_alpha=0.0,
        line_width=1.0,
        line_style="-",
        zorder=2,
    ),
    MapLayer.WALKWAY: PlotConfig(
        fill_color=Color("#d4d19e"),
        fill_color_alpha=1.0,
        line_color=Color("#d4d19e"),
        line_color_alpha=0.0,
        line_width=1.0,
        line_style="-",
        zorder=1,
    ),
    MapLayer.CARPARK: PlotConfig(
        fill_color=Color("#b9d3b4"),
        fill_color_alpha=1.0,
        line_color=Color("#b9d3b4"),
        line_color_alpha=0.0,
        line_width=0.0,
        line_style="-",
        zorder=1,
    ),
    MapLayer.GENERIC_DRIVABLE: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=1.0,
        line_style="-",
        zorder=1,
    ),
}

BOX_DETECTION_CONFIG: Dict[DefaultBoxDetectionLabel, PlotConfig] = {
    DefaultBoxDetectionLabel.VEHICLE: PlotConfig(
        fill_color=ELLIS_5[4],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=HEADING_MARKER_STYLE,
        marker_size=1.0,
        zorder=3,
    ),
    DefaultBoxDetectionLabel.PEDESTRIAN: PlotConfig(
        fill_color=NEW_TAB_10[6],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=None,
        marker_size=1.0,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.BICYCLE: PlotConfig(
        fill_color=ELLIS_5[3],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=HEADING_MARKER_STYLE,
        marker_size=1.0,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.TRAFFIC_CONE: PlotConfig(
        fill_color=NEW_TAB_10[5],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.BARRIER: PlotConfig(
        fill_color=NEW_TAB_10[5],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.CZONE_SIGN: PlotConfig(
        fill_color=NEW_TAB_10[5],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.GENERIC_OBJECT: PlotConfig(
        fill_color=NEW_TAB_10[5],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.SIGN: PlotConfig(
        fill_color=NEW_TAB_10[8],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.EGO: PlotConfig(
        fill_color=ELLIS_5[0],
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="-",
        marker_style=HEADING_MARKER_STYLE,
        zorder=4,
    ),
}

EGO_VEHICLE_CONFIG: PlotConfig = PlotConfig(
    fill_color=ELLIS_5[0],
    fill_color_alpha=1.0,
    line_color=BLACK,
    line_color_alpha=1.0,
    line_width=1.0,
    line_style="-",
    marker_style=HEADING_MARKER_STYLE,
    zorder=4,
)

CENTERLINE_CONFIG: PlotConfig = PlotConfig(
    fill_color=WHITE,
    fill_color_alpha=1.0,
    line_color=DARKER_GREY,
    line_color_alpha=1.0,
    line_width=1.0,
    line_style="--",
    zorder=3,
)
ROUTE_CONFIG: PlotConfig = PlotConfig(
    fill_color=Color("#f2c6c0ff"),
    fill_color_alpha=1.0,
    line_color=Color("#f2c6c0ff"),
    line_color_alpha=0.0,
    line_width=1.0,
    line_style="-",
    zorder=2,
)

MARK_CONFIG: PlotConfig = PlotConfig(
    fill_color=TAB_10[6],
    fill_color_alpha=1.0,
    line_color=TAB_10[6],
    line_color_alpha=1.0,
    line_width=1.0,
    line_style="-",
    marker_style=HEADING_MARKER_STYLE,
    marker_size=1.0,
    zorder=3,
)


TRAFFIC_LIGHT_CONFIG: Dict[TrafficLightStatus, PlotConfig] = {
    TrafficLightStatus.RED: PlotConfig(
        line_color=TAB_10[3],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.YELLOW: PlotConfig(
        line_color=TAB_10[1],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.GREEN: PlotConfig(
        line_color=TAB_10[2],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.OFF: PlotConfig(
        line_color=TAB_10[5],
        line_color_alpha=1.0,
        line_width=1.0,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.UNKNOWN: PlotConfig(
        line_color=TAB_10[5],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
}
