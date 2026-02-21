from typing import Callable, Final

# Modular log directory file names
# Each log is stored as a folder containing one Arrow file per modality.
# The index.arrow file is the central "spine" containing uuid + timestamp + metadata.

INDEX_FILE: Final[str] = "index.arrow"

# Non-sensor modality files
EGO_STATE_FILE: Final[str] = "EgoState.arrow"
BOX_DETECTIONS_FILE: Final[str] = "BoxDetections.arrow"
TRAFFIC_LIGHTS_FILE: Final[str] = "TrafficLights.arrow"
SCENARIO_TAGS_FILE: Final[str] = "ScenarioTags.arrow"
ROUTE_FILE: Final[str] = "Route.arrow"

# Sensor modality files (parameterized by sensor name)
PINHOLE_CAMERA_FILE: Callable[[str], str] = lambda name: f"PinholeCamera.{name}.arrow"
FISHEYE_CAMERA_FILE: Callable[[str], str] = lambda name: f"FisheyeCamera.{name}.arrow"
LIDAR_FILE: Callable[[str], str] = lambda name: f"LiDAR.{name}.arrow"
