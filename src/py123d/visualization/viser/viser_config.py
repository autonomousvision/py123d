from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from py123d.datatypes.sensors.camera.pinhole_camera import PinholeCameraType
from py123d.datatypes.sensors.lidar.lidar import LiDARType

all_camera_types: List[PinholeCameraType] = [
    PinholeCameraType.CAM_F0,
    PinholeCameraType.CAM_B0,
    PinholeCameraType.CAM_L0,
    PinholeCameraType.CAM_L1,
    PinholeCameraType.CAM_L2,
    PinholeCameraType.CAM_R0,
    PinholeCameraType.CAM_R1,
    PinholeCameraType.CAM_R2,
]

all_lidar_types: List[LiDARType] = [
    LiDARType.LIDAR_MERGED,
    LiDARType.LIDAR_TOP,
    LiDARType.LIDAR_FRONT,
    LiDARType.LIDAR_SIDE_LEFT,
    LiDARType.LIDAR_SIDE_RIGHT,
    LiDARType.LIDAR_BACK,
    LiDARType.LIDAR_DOWN,
]


@dataclass
class ViserConfig:

    # Server
    server_host: str = "localhost"
    server_port: int = 8080
    server_label: str = "123D Viser Server"
    server_verbose: bool = True

    # Theme
    theme_control_layout: Literal["floating", "collapsible", "fixed"] = "floating"
    theme_control_width: Literal["small", "medium", "large"] = "large"
    theme_dark_mode: bool = False
    theme_show_logo: bool = True
    theme_show_share_button: bool = True
    theme_brand_color: Optional[Tuple[int, int, int]] = None

    # Map
    map_visible: bool = True
    map_radius: float = 100.0  # [m]
    map_non_road_z_offset: float = 0.1  # small z-translation to place crosswalks, parking, etc. on top of the road
    map_requery: bool = True  # Re-query map when ego vehicle moves out of current map bounds

    # Bounding boxes
    bounding_box_visible: bool = True
    bounding_box_type: Literal["mesh", "lines"] = "mesh"
    bounding_box_line_width: float = 4.0

    # Cameras
    # -> Frustum
    camera_frustum_visible: bool = True
    camera_frustum_types: List[PinholeCameraType] = field(default_factory=lambda: all_camera_types.copy())
    camera_frustum_frustum_scale: float = 1.0
    camera_frustum_image_scale: float = 0.25  # Resize factor for the camera image shown on the frustum (<1.0 for speed)

    # -> GUI
    camera_gui_visible: bool = True
    camera_gui_types: List[PinholeCameraType] = field(default_factory=lambda: [PinholeCameraType.CAM_F0].copy())
    camera_gui_image_scale: float = 0.25  # Resize factor for the camera image shown in the GUI (<1.0 for speed)

    # LiDAR
    lidar_visible: bool = True
    lidar_types: List[LiDARType] = field(default_factory=lambda: all_lidar_types.copy())
    lidar_point_size: float = 0.05
    lidar_point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = "circle"
