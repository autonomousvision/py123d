from typing import Union

from py123d.datatypes.sensors.camera.fisheye_mei_camera import FisheyeMEICameraType
from py123d.datatypes.sensors.camera.pinhole_camera import PinholeCameraType


def get_camera_type_by_value(value: int) -> Union[PinholeCameraType, FisheyeMEICameraType]:
    """Dynamically determine camera type based on value range."""
    pinhole_values = [member.value for member in PinholeCameraType]
    fisheye_values = [member.value for member in FisheyeMEICameraType]

    if value in pinhole_values:
        return PinholeCameraType(value)
    elif value in fisheye_values:
        return FisheyeMEICameraType(value)
    else:
        raise ValueError(
            f"Invalid camera type value: {value}. "
            f"Valid PinholeCameraType values: {pinhole_values}, "
            f"Valid FisheyeMEICameraType values: {fisheye_values}"
        )


def deserialize_camera_type(camera_str: str) -> Union[PinholeCameraType, FisheyeMEICameraType]:
    """Deserialize camera type string to appropriate enum."""
    try:
        return PinholeCameraType.deserialize(camera_str)
    except (ValueError, KeyError):
        pass

    try:
        return FisheyeMEICameraType.deserialize(camera_str)
    except (ValueError, KeyError):
        pass

    pinhole_names = [member.name.lower() for member in PinholeCameraType]
    fisheye_names = [member.name.lower() for member in FisheyeMEICameraType]
    raise ValueError(
        f"Unknown camera type: '{camera_str}'. "
        f"Valid PinholeCameraType names: {pinhole_names}, "
        f"Valid FisheyeMEICameraType names: {fisheye_names}"
    )
