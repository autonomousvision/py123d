from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.sensors.base_camera import ALL_FISHEYE_MEI_CAMERA_IDS, ALL_PINHOLE_CAMERA_IDS, CameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.visualization.color.color import ELLIS_5


def _resolve_enum_arguments(
    serial_enum_cls: SerialIntEnum, input: Optional[List[Union[int, str, SerialIntEnum]]]
) -> Optional[List[SerialIntEnum]]:
    if input is None:
        return None
    if not isinstance(input, list):
        raise TypeError(f"input must be a list of {serial_enum_cls.__name__}, got {type(input)}")
    return [serial_enum_cls.from_arbitrary(value) for value in input]


@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 8080
    label: str = "123D Viser Server"
    verbose: bool = True


@dataclass
class ThemeConfig:
    control_layout: Literal["floating", "collapsible", "fixed"] = "floating"
    control_width: Literal["small", "medium", "large"] = "large"
    dark_mode: bool = False
    show_logo: bool = True
    show_share_button: bool = True
    brand_color: Optional[Tuple[int, int, int]] = ELLIS_5[0].rgb


@dataclass
class PlaybackConfig:
    is_playing: bool = False
    speed: float = 1.0


@dataclass
class MapConfig:
    visible: bool = True
    radius: float = 200.0
    non_road_z_offset: float = 0.1
    opacity: float = 1.0
    requery: bool = True
    visible_layers: List[MapLayer] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.visible_layers = _resolve_enum_arguments(MapLayer, self.visible_layers)


@dataclass
class DetectionConfig:
    visible: bool = True
    type: Literal["mesh", "lines", "mesh+lines"] = "mesh+lines"
    line_width: float = 4.0
    opacity: float = 0.5


@dataclass
class CameraFrustumConfig:
    visible: bool = True
    pinhole_types: List[CameraID] = field(default_factory=lambda: ALL_PINHOLE_CAMERA_IDS.copy())
    fisheye_types: List[CameraID] = field(default_factory=lambda: ALL_FISHEYE_MEI_CAMERA_IDS.copy())
    scale: float = 1.0
    image_scale: int = 4
    fisheye_fov: float = 185.0

    def __post_init__(self):
        self.pinhole_types = _resolve_enum_arguments(CameraID, self.pinhole_types)
        self.fisheye_types = _resolve_enum_arguments(CameraID, self.fisheye_types)
        self.image_scale = int(self.image_scale)


@dataclass
class CameraGuiConfig:
    visible: bool = True
    types: List[CameraID] = field(default_factory=lambda: [CameraID.PCAM_F0])
    image_scale: int = 4

    def __post_init__(self):
        self.types = _resolve_enum_arguments(CameraID, self.types)
        self.image_scale = int(self.image_scale)


@dataclass
class LidarConfig:
    visible: bool = True
    ids: List[LidarID] = field(default_factory=lambda: [LidarID.LIDAR_MERGED])
    point_size: float = 0.03
    point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = "circle"
    point_color: Literal["none", "distance", "ids", "intensity", "channel", "timestamps", "range", "elongation"] = (
        "none"
    )
    stride_step: int = 1

    def __post_init__(self):
        self.ids = _resolve_enum_arguments(LidarID, self.ids)


_SUB_CONFIG_FIELDS = {
    "server": ServerConfig,
    "theme": ThemeConfig,
    "playback": PlaybackConfig,
    "map": MapConfig,
    "detection": DetectionConfig,
    "camera_frustum": CameraFrustumConfig,
    "camera_gui": CameraGuiConfig,
    "lidar": LidarConfig,
}


@dataclass
class ViserConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    map: MapConfig = field(default_factory=MapConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera_frustum: CameraFrustumConfig = field(default_factory=CameraFrustumConfig)
    camera_gui: CameraGuiConfig = field(default_factory=CameraGuiConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)

    def __post_init__(self):
        # Hydra instantiate with _convert_='all' produces plain dicts for nested configs.
        # Convert them to the proper dataclass types.
        for field_name, config_cls in _SUB_CONFIG_FIELDS.items():
            value = getattr(self, field_name)
            if isinstance(value, dict):
                setattr(self, field_name, config_cls(**value))
