from typing import Dict, List

from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType
from d123.datatypes.sensors.lidar.lidar import LiDARType

WOPD_AVAILABLE_SPLITS: List[str] = [
    "wopd_train",
    "wopd_val",
    "wopd_test",
]

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto#L63
WOPD_DETECTION_NAME_DICT: Dict[int, DetectionType] = {
    0: DetectionType.GENERIC_OBJECT,  # TYPE_UNKNOWN
    1: DetectionType.VEHICLE,  # TYPE_VEHICLE
    2: DetectionType.PEDESTRIAN,  # TYPE_PEDESTRIAN
    3: DetectionType.SIGN,  # TYPE_SIGN
    4: DetectionType.BICYCLE,  # TYPE_CYCLIST
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L50
WOPD_CAMERA_TYPES: Dict[int, PinholeCameraType] = {
    1: PinholeCameraType.CAM_F0,  # front_camera
    2: PinholeCameraType.CAM_L0,  # front_left_camera
    3: PinholeCameraType.CAM_R0,  # front_right_camera
    4: PinholeCameraType.CAM_L1,  # left_camera
    5: PinholeCameraType.CAM_R1,  # right_camera
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L66
WOPD_LIDAR_TYPES: Dict[int, LiDARType] = {
    0: LiDARType.LIDAR_UNKNOWN,  # UNKNOWN
    1: LiDARType.LIDAR_TOP,  # TOP
    2: LiDARType.LIDAR_FRONT,  # FRONT
    3: LiDARType.LIDAR_SIDE_LEFT,  # SIDE_LEFT
    4: LiDARType.LIDAR_SIDE_RIGHT,  # SIDE_RIGHT
    5: LiDARType.LIDAR_BACK,  # REAR
}
