"""Constants and mappings for NUREC dataset."""

from typing import Dict, Final

from py123d.datatypes.detections.detection_types import DetectionType
from py123d.datatypes.sensors.camera.pinhole_camera import PinholeCameraType

# Dataset splits
# I'm not sure how to handle it best, as technically the batches can be
# considered to be separate splits, but then we would have a lot of splits...
# Also, it is not really clear to me why some batches don't exist, e.g.
# Batch0001
NUREC_DATA_SPLITS: Final[list[str]] = [
    "nurec_batch0002",
    "nurec_batch0005",
]

# Camera name mapping from NUREC to py123d camera types
# Note: Nurec provided camera information for multiple setups, but only the mp4
# file for a single one
NUREC_CAMERA_MAPPING: Final[Dict[str, PinholeCameraType]] = {
    "camera_front_wide_120fov": PinholeCameraType.CAM_F0,
    "camera_front_tele_30fov": PinholeCameraType.CAM_F0,  # Alternative front camera
    "camera_cross_left_120fov": PinholeCameraType.CAM_L0,
    "camera_cross_right_120fov": PinholeCameraType.CAM_R0,
    "camera_rear_left_70fov": PinholeCameraType.CAM_L2,
    "camera_rear_right_70fov": PinholeCameraType.CAM_R2,
}

# Object class mapping from NUREC to py123d detection types
# TODO There are also some classes I haven't included yet, e.g. animals
NUREC_TO_DETECTION_TYPE: Final[Dict[str, DetectionType]] = {
    "automobile": DetectionType.VEHICLE,
    "heavy_truck": DetectionType.VEHICLE,
    "trailer": DetectionType.VEHICLE,
    "person": DetectionType.PEDESTRIAN,
    "protruding_object": DetectionType.GENERIC_OBJECT,
}

# Default bounding box dimensions [length, width, height] in meters for each class
# These are reasonable estimates since NUREC doesn't provide actual dimensions
NUREC_DEFAULT_BOX_DIMENSIONS: Final[Dict[str, tuple[float, float, float]]] = {
    "automobile": (4.5, 2.0, 1.5),  # Typical sedan/SUV
    "heavy_truck": (8.0, 2.5, 3.0),  # Large truck
    "trailer": (6.0, 2.5, 2.5),  # Trailer
    "person": (0.5, 0.5, 1.7),  # Average person
    "protruding_object": (1.0, 1.0, 1.0),  # Generic small object
}

# Target sampling rate in seconds
NUREC_TARGET_DT: Final[float] = 0.1  # 10 Hz sampling

# NUREC uses microsecond timestamps
NUREC_TIMESTAMP_UNIT: Final[str] = "microseconds"
