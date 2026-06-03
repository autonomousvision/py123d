from typing import Union

from py123d.datatypes.sensors.base_camera import (
    BaseCameraMetadata,
    Camera,
    CameraChannelType,
    CameraID,
    CameraModel,
    camera_metadata_from_dict,
)
from py123d.datatypes.sensors.camera_segmentation_label import (
    CAMERA_SEGMENTATION_LABEL_REGISTRY,
    CameraSegmentationLabel,
    DefaultCameraSegmentationLabel,
    register_camera_segmentation_label,
)
from py123d.datatypes.sensors.fisheye_mei_camera import (
    FisheyeMEICameraMetadata,
    FisheyeMEIDistortion,
    FisheyeMEIDistortionIndex,
    FisheyeMEIProjection,
    FisheyeMEIProjectionIndex,
)
from py123d.datatypes.sensors.ftheta_camera import (
    FThetaCameraMetadata,
    FThetaIntrinsics,
    FThetaIntrinsicsIndex,
)
from py123d.datatypes.sensors.lidar import (
    Lidar,
    LidarFeature,
    LidarID,
    LidarMergedMetadata,
    LidarMetadata,
)
from py123d.datatypes.sensors.lidar_segmentation_label import (
    LIDAR_SEGMENTATION_LABEL_REGISTRY,
    DefaultLidarSegmentationLabel,
    LidarSegmentationLabel,
    register_lidar_segmentation_label,
)
from py123d.datatypes.sensors.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeDistortionIndex,
    PinholeIntrinsics,
    PinholeIntrinsicsIndex,
)
from py123d.datatypes.sensors.segmentation_camera import SegmentationCameraMetadata

CameraMetadata = Union[PinholeCameraMetadata, FisheyeMEICameraMetadata, FThetaCameraMetadata]

__all__ = [
    # Base camera
    "BaseCameraMetadata",
    "Camera",
    "CameraChannelType",
    "CameraID",
    "CameraModel",
    "camera_metadata_from_dict",
    # Camera segmentation
    "SegmentationCameraMetadata",
    "CameraSegmentationLabel",
    "DefaultCameraSegmentationLabel",
    "CAMERA_SEGMENTATION_LABEL_REGISTRY",
    "register_camera_segmentation_label",
    # Lidar segmentation
    "LidarSegmentationLabel",
    "DefaultLidarSegmentationLabel",
    "LIDAR_SEGMENTATION_LABEL_REGISTRY",
    "register_lidar_segmentation_label",
    # Fisheye MEI camera
    "FisheyeMEICameraMetadata",
    "FisheyeMEIDistortion",
    "FisheyeMEIDistortionIndex",
    "FisheyeMEIProjection",
    "FisheyeMEIProjectionIndex",
    # F-theta camera
    "FThetaCameraMetadata",
    "FThetaIntrinsics",
    "FThetaIntrinsicsIndex",
    # Lidar
    "Lidar",
    "LidarFeature",
    "LidarID",
    "LidarMergedMetadata",
    "LidarMetadata",
    # Pinhole camera
    "PinholeCameraMetadata",
    "PinholeDistortion",
    "PinholeDistortionIndex",
    "PinholeIntrinsics",
    "PinholeIntrinsicsIndex",
    # Type alias
    "CameraMetadata",
]
