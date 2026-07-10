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
from py123d.datatypes.sensors.depth_camera import (
    DEPTH_BITS,
    DepthCameraMetadata,
    colorize_depth_map,
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
from py123d.datatypes.sensors.radar import (
    RADAR_FEATURE_DTYPES,
    Radar,
    RadarFeature,
    RadarID,
    RadarMergedMetadata,
    RadarMetadata,
    get_individual_radar,
    get_merged_radar,
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
    # Camera depth
    "DepthCameraMetadata",
    "DEPTH_BITS",
    "colorize_depth_map",
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
    # Radar
    "Radar",
    "RadarFeature",
    "RADAR_FEATURE_DTYPES",
    "RadarID",
    "RadarMergedMetadata",
    "RadarMetadata",
    "get_individual_radar",
    "get_merged_radar",
    # Pinhole camera
    "PinholeCameraMetadata",
    "PinholeDistortion",
    "PinholeDistortionIndex",
    "PinholeIntrinsics",
    "PinholeIntrinsicsIndex",
    # Type alias
    "CameraMetadata",
]
