from dataclasses import dataclass
from typing import List, Optional, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType

# TODO: Add more filter options (e.g. scene tags, ego movement, or whatever appropriate)


@dataclass
class SceneFilter:
    """Class to filter scenes when building scenes from logs."""

    split_types: Optional[List[str]] = None
    """List of split types to filter scenes by (e.g. `train`, `val`, `test`)."""

    split_names: Optional[List[str]] = None
    """List of split names to filter scenes by (in the form `{dataset_name}_{split_type}`)."""

    log_names: Optional[List[str]] = None
    """Name of logs to include scenes from."""

    locations: Optional[List[str]] = None
    """List of locations to filter scenes by."""

    scene_uuids: Optional[List[str]] = None
    """List of scene UUIDs to include."""

    timestamp_threshold_s: Optional[float] = None
    """Minimum time between the start timestamps of two consecutive scenes."""

    duration_s: Optional[float] = 10.0
    """Duration of each scene in seconds."""

    history_s: Optional[float] = 3.0
    """History duration of each scene in seconds."""

    pinhole_camera_types: Optional[List[PinholeCameraType]] = None
    """List of :class:`PinholeCameraType` to include in the scenes."""

    fisheye_mei_camera_types: Optional[List[FisheyeMEICameraType]] = None
    """List of :class:`FisheyeMEICameraType` to include in the scenes."""

    max_num_scenes: Optional[int] = None
    """Maximum number of scenes to return."""

    shuffle: bool = False
    """Whether to shuffle the returned scenes."""

    def __post_init__(self):
        def _resolve_enum_arguments(
            serial_enum_cls: SerialIntEnum,
            input: Optional[List[Union[int, str, SerialIntEnum]]],
        ):
            if input is None:
                return None
            return [serial_enum_cls.from_arbitrary(value) for value in input]

        self.pinhole_camera_types = _resolve_enum_arguments(PinholeCameraType, self.pinhole_camera_types)
        self.fisheye_mei_camera_types = _resolve_enum_arguments(FisheyeMEICameraType, self.fisheye_mei_camera_types)
