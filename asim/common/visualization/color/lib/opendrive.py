# TODO: Add IntEnum
from typing import Dict

from asim.common.visualization.color.color import Color

# TODO: fix these colors
OPENDRIVE_LANE_TYPES: Dict[str, Color] = {
    "biking": Color("#cf102d"),
    "border": Color("#a55e37"),
    "connectingRamp": Color("#a8d300"),
    "curb": Color("#9778d3"),
    "driving": Color("#efd7ab"),
    # "driving +": Color("#efd7ab"),
    # "driving -": Color("#cdd8e8"),
    "entry": Color("#ead960"),
    "exit": Color("#6799cc"),
    "median": Color("#7c5447"),
    "none": Color("#939598"),
    "offRamp": Color("#2379b9"),
    "onRamp": Color("#ffd402"),
    "parking": Color("#62269e"),
    "rail": Color("#382bb2"),
    "restricted": Color("#ff671b"),
    "shoulder": Color("#006241"),
    "walking": Color("#79242f"),
    "sidewalk": Color("#79242f"),  # Used in CARLA maps
    "slipLane": Color("#00945e"),
    "stop": Color("#92d5ac"),
    "tram": Color("#6d6de2"),
}

# TODO: fix these colors
OPENDRIVE_LANE_CUSTOM: Dict[str, Color] = {
    "biking": Color("#0ea5deff"),
    "border": Color("#ffb812ff"),
    "connectingRamp": Color("#a8d300"),
    "curb": Color("#9778d3"),
    "driving": Color("#b3b3b3ff"),
    # "driving +": Color("#efd7ab"),
    # "driving -": Color("#cdd8e8"),
    "entry": Color("#ead960"),
    "exit": Color("#6799cc"),
    "median": Color("#6d6de2"),
    "none": Color("#fc0303ff"),
    "offRamp": Color("#2379b9"),
    "onRamp": Color("#ffd402"),
    "parking": Color("#3dd976ff"),
    "rail": Color("#382bb2"),
    "restricted": Color("#ff671b"),
    "shoulder": Color("#33a2e2ff"),
    # "walking": Color("#79242f"),
    "sidewalk": Color("#b060b0ff"),
    "slipLane": Color("#00945e"),
    "stop": Color("#92d5ac"),
    "tram": Color("#6d6de2"),
    "bidirectional": Color("#727272ff"),  # Used in CARLA maps
}
