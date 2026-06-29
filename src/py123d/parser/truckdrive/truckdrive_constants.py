"""Constants for the public TruckDrive dataset integration."""

from __future__ import annotations

from typing import Dict, Tuple

from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.sensors.lidar import LidarID

DATASET_NAME = "truckdrive"

CLOUDFRONT_BASE_URL = "https://d3ehgyu1hepsur.cloudfront.net"
S3_PREFIX = "TruckDrive/"

DEFAULT_MODALITIES: Tuple[str, ...] = (
    "camera",
    "lidar",
    "poses",
    "calibrations",
    "annotations",
)

MODALITY_ZIP_FILES: Dict[str, str] = {
    "camera": "camera.zip",
    "lidar": "lidar.zip",
    "poses": "poses.zip",
    "calibrations": "calibrations.zip",
    "annotations": "annotations.zip",
    "radar": "radar.zip",
    "accumulated_gt_depth": "accumulated_gt_depth.zip",
}

# Superset of camera names documented by the TruckDrive devkit. Discover on disk per scene.
CAMERA_NAMES: Tuple[str, ...] = (
    "forward_center_medium",
    "forward_left_medium",
    "forward_left_narrow",
    "forward_left_wide",
    "forward_right_medium",
    "forward_right_narrow",
    "forward_right_wide",
    "rearward_left_bottom_medium",
    "rearward_left_top_medium",
    "rearward_right_bottom_medium",
    "rearward_right_top_medium",
    "sideward_left_front_wide",
    "sideward_left_back_wide",
    "sideward_right_front_wide",
    "sideward_right_back_wide",
)

OUSTER_LIDAR_NAMES: Tuple[str, ...] = (
    "forward_center",
    "sideward_left",
    "sideward_right",
)

AEVA_JOINT_LIDAR_REL_PATH = "lidar/aeva/joint_lidars/points"

# Frame names in calibrations/calib_tf_tree_full.json
VEHICLE_FRAME = "vehicle"
VELODYNE_FRAME = "velodyne"
AEVA_REFERENCE_LIDAR_FRAME = "lidar_aeva_forward_center_wide"

# TODO(truckdrive): confirm tractor wheelbase and vehicle dimensions with Torc.
DEFAULT_WHEEL_BASE_M = 6.0
DEFAULT_VEHICLE_WIDTH_M = 2.6
DEFAULT_VEHICLE_LENGTH_M = 22.0
DEFAULT_VEHICLE_HEIGHT_M = 4.0
DEFAULT_VEHICLE_NAME = "freightliner_cascadia"

LABEL_MAPPING: Dict[str, int] = {'Animal': 2,
 'DelineatorGroupDontCare': -1,
 'DynamicTrafficSign': 7,
 'GeneralTrafficSign': 7,
 'GeneralTrafficSign-Regulatory': 7,
 'GeneralTrafficSign-School': 7,
 'LaneUseSignal': 7,
 'LaneUseSignal-Green': 7,
 'LaneUseSignal-Off': 7,
 'LaneUseSignal-Red': 7,
 'OutOfLidarRangeVehicleGroup': -1,
 'ParkingLotVehicleGroup': -1,
 'Person': 2,
 'Person-Other': 2,
 'Person-Pedestrian': 2,
 'Person-Rider': 2,
 'Person-Skater': 2,
 'Person-TrafficControl': 2,
 'RoadDebris': 3,
 'RoadDebris-Other': 3,
 'RoadDebris-Pothole': 3,
 'RoadDebris-RoadKill': 3,
 'RoadDebris-Tire': 3,
 'RoadDebris-Vegetation': 3,
 'RoadObstruction': 3,
 'RoadObstruction-Barrel': 3,
 'RoadObstruction-Barricade': 3,
 'RoadObstruction-Cone': 3,
 'RoadObstruction-Delineator': 3,
 'RoadObstruction-Other': 3,
 'RoadObstruction-VerticalPanel': 3,
 'TrafficSign': 7,
 'TrafficSign-Advisory': 7,
 'TrafficSign-Informational': 7,
 'TrafficSign-LaneEnds-Right': 7,
 'TrafficSign-Merge': 7,
 'TrafficSign-Merge-Right': 7,
 'TrafficSign-Other': 7,
 'TrafficSign-RoadworkAhead': 7,
 'TrafficSign-SpeedLimit': 7,
 'TrafficSign-SpeedLimit-Exit': 7,
 'TrafficSign-SpeedLimit-HardLimit': 7,
 'TrafficSign-SpeedLimit-Upcoming': 7,
 'TrafficSign-Yield': 7,
 'TrafficSignal': 7,
 'TrafficSignal-GreenForwardArrow': 7,
 'TrafficSignal-GreenLeftArrow': 7,
 'TrafficSignal-GreenRightArrow': 7,
 'TrafficSignal-GreenSolid': 7,
 'TrafficSignal-MultiState': 7,
 'TrafficSignal-Off': 7,
 'TrafficSignal-Other': 7,
 'TrafficSignal-RedLeftArrow': 7,
 'TrafficSignal-RedRightArrow': 7,
 'TrafficSignal-RedSolid': 7,
 'TrafficSignal-YellowLeftArrow': 7,
 'TrafficSignal-YellowRightArrow': 7,
 'TrafficSignal-YellowSolid': 7,
 'VRUvehicle': 6,
 'VRUvehicle-Bicycle': 0,
 'VRUvehicle-Motorcycle': 0,
 'VRUvehicle-Other': 6,
 'VRUvehicle-StandingScooter': 0,
 'VRUvehicle-Trailer': 6,
 'VRUvehicle-Wheelchair': 0,
 'Vehicle': 6,
 'Vehicle-Bicycle': 0,
 'Vehicle-Bus': 6,
 'Vehicle-DeliveryVan': 6,
 'Vehicle-EgoVehicle-Cab': -1,
 'Vehicle-EgoVehicle-Trailer': -1,
 'Vehicle-Emergency': 8,
 'Vehicle-Equipment': 6,
 'Vehicle-HeavyVehicle': 6,
 'Vehicle-Motorcycle': 0,
 'Vehicle-Other': 6,
 'Vehicle-Passenger': 1,
 'Vehicle-Police': 8,
 'Vehicle-RV': 6,
 'Vehicle-SchoolBus': 6,
 'Vehicle-SemiTruck-Cab': 4,
 'Vehicle-SemiTruck-Trailer': 5,
 'Vehicle-SingleUnitTruck': 6,
 'Vehicle-Trailer': 6,
 'Vehicle-Unibody': 6,
 'WalkSignal': 7}

MAPPED_CATEGORIES: Dict[str, int] = {'DontCare': -1,
 'Bike': 0,
 'Passenger-Car': 1,
 'Person': 2,
 'RoadObstruction': 3,
 'SemiTruck-Cab': 4,
 'SemiTruck-Trailer': 5,
 'Vehicle': 6,
 'TrafficSign': 7,
 'EmergencyVehicle': 8}

# TODO(truckdrive): add new val and test scenes
VAL_SCENES: Tuple[str, ...] = ('scene_28_1', 'scene_28_2', 'scene_28_3', 'scene_28_4', 'scene_28_5', 'scene_28_6', 'scene_28_7', 'scene_28_8', 'scene_28_9', 'scene_28_10', 'scene_28_11', 'scene_28_12', 'scene_28_13', 'scene_28_14', 'scene_28_15', 'scene_28_16', 'scene_28_17', 'scene_28_18', 'scene_28_19', 'scene_28_20', 'scene_28_21', 'scene_28_22', 'scene_28_23', 'scene_28_24')

CAMERA_ID_MAPPING: Dict[str, CameraID] = {
    "forward_center_medium": CameraID.PCAM_F0,
    "forward_left_medium": CameraID.PCAM_L0,
    "forward_left_narrow": CameraID.FTCAM_TELE_F0,
    "forward_left_wide": CameraID.PCAM_L1,
    "forward_right_medium": CameraID.PCAM_R0,
    "forward_right_narrow": CameraID.FTCAM_TELE_B0,
    "forward_right_wide": CameraID.PCAM_R2,
    "rearward_left_bottom_medium": CameraID.PCAM_L2,
    "rearward_left_top_medium": CameraID.FTCAM_L0,
    "rearward_right_bottom_medium": CameraID.PCAM_B0,
    "rearward_right_top_medium": CameraID.FTCAM_R0,
    "sideward_left_front_wide": CameraID.FTCAM_L1,
    "sideward_left_back_wide": CameraID.FTCAM_R1,
    "sideward_right_front_wide": CameraID.FTCAM_R0,
    "sideward_right_back_wide": CameraID.FTCAM_R1,
}

OUSTER_LIDAR_ID_MAPPING: Dict[str, LidarID] = {
    "forward_center": LidarID.LIDAR_FRONT,
    "sideward_left": LidarID.LIDAR_SIDE_LEFT,
    "sideward_right": LidarID.LIDAR_SIDE_RIGHT,
}

AEVA_LIDAR_ID = LidarID.LIDAR_MERGED

# TODO(truckdrive): add new test scenes
TRUCKDRIVE_SPLITS: Tuple[str, ...] = ("truckdrive_val",)
