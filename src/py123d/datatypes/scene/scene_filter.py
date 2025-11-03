from dataclasses import dataclass
from typing import List, Optional, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType

# TODO: Add more filter options (e.g. scene tags, ego movement, or whatever appropriate)


@dataclass
class SceneFilter:

    split_types: Optional[List[str]] = None
    split_names: Optional[List[str]] = None
    log_names: Optional[List[str]] = None

    locations: Optional[List[str]] = None  # TODO:
    scene_uuids: Optional[List[str]] = None  # TODO:

    timestamp_threshold_s: Optional[float] = None  # TODO:
    ego_displacement_minimum_m: Optional[float] = None  # TODO:

    duration_s: Optional[float] = 10.0
    history_s: Optional[float] = 3.0

    pinhole_camera_types: Optional[List[PinholeCameraType]] = None
    fisheye_mei_camera_types: Optional[List[FisheyeMEICameraType]] = None

    max_num_scenes: Optional[int] = None
    shuffle: bool = False

    def __post_init__(self):
        def _resolve_enum_arguments(
            serial_enum_cls: SerialIntEnum, input: Optional[List[Union[int, str, SerialIntEnum]]]
        ) -> List[SerialIntEnum]:

            if input is None:
                return None
            assert isinstance(input, list), f"input must be a list of {serial_enum_cls.__name__}"
            return [serial_enum_cls.from_arbitrary(value) for value in input]

        self.pinhole_camera_types = _resolve_enum_arguments(PinholeCameraType, self.pinhole_camera_types)
        self.fisheye_mei_camera_types = _resolve_enum_arguments(FisheyeMEICameraType, self.fisheye_mei_camera_types)
