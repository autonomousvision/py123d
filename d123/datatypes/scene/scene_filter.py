from dataclasses import dataclass
from typing import List, Optional

from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType

# TODO: Add more filter options (e.g. scene tags, ego movement, or whatever appropriate)


@dataclass
class SceneFilter:

    split_types: Optional[List[str]] = None
    split_names: Optional[List[str]] = None
    # scene_tags: List[str] = None
    log_names: Optional[List[str]] = None

    map_names: Optional[List[str]] = None  # TODO:
    scene_uuids: Optional[List[str]] = None  # TODO:

    timestamp_threshold_s: Optional[float] = None  # TODO:
    ego_displacement_minimum_m: Optional[float] = None  # TODO:

    duration_s: Optional[float] = 10.0
    history_s: Optional[float] = 3.0

    camera_types: Optional[List[PinholeCameraType]] = None

    max_num_scenes: Optional[int] = None
    shuffle: bool = False

    def __post_init__(self):
        if self.camera_types is not None:
            assert isinstance(self.camera_types, list), "camera_types must be a list of CameraType"
            camera_types = []
            for camera_type in self.camera_types:
                if isinstance(camera_type, str):
                    camera_type = PinholeCameraType.deserialize[camera_type]
                    camera_types.append(camera_type)
                elif isinstance(camera_type, int):
                    camera_type = PinholeCameraType(camera_type)
                    camera_types.append(camera_type)
                else:
                    raise ValueError(f"Invalid camera type: {camera_type}")
            self.camera_types = camera_types
