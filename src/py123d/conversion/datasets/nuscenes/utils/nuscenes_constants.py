import os
from pathlib import Path
from typing import Final, List

from py123d.datatypes.detections.box_detection_types import BoxDetectionType
from py123d.datatypes.sensors.camera.pinhole_camera import PinholeCameraType

NUSCENES_MAPS: List[str] = ["boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"]

NUSCENES_DATA_SPLITS: Final[List[str]] = [
    "nuscenes_train",
    "nuscenes_val",
    "nuscenes_test",
    "nuscenes-mini_train",
    "nuscenes-mini_val",
]

TARGET_DT: Final[float] = 0.1
NUSCENES_DT: Final[float] = 0.5
SORT_BY_TIMESTAMP: Final[bool] = True
NUSCENES_DETECTION_NAME_DICT = {
    # Vehicles (4+ wheels)
    "vehicle.car": BoxDetectionType.VEHICLE,
    "vehicle.truck": BoxDetectionType.VEHICLE,
    "vehicle.bus.bendy": BoxDetectionType.VEHICLE,
    "vehicle.bus.rigid": BoxDetectionType.VEHICLE,
    "vehicle.construction": BoxDetectionType.VEHICLE,
    "vehicle.emergency.ambulance": BoxDetectionType.VEHICLE,
    "vehicle.emergency.police": BoxDetectionType.VEHICLE,
    "vehicle.trailer": BoxDetectionType.VEHICLE,
    # Bicycles / Motorcycles
    "vehicle.bicycle": BoxDetectionType.BICYCLE,
    "vehicle.motorcycle": BoxDetectionType.BICYCLE,
    # Pedestrians (all subtypes)
    "human.pedestrian.adult": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.child": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.construction_worker": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.personal_mobility": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.police_officer": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.stroller": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.wheelchair": BoxDetectionType.PEDESTRIAN,
    # Traffic cone / barrier
    "movable_object.trafficcone": BoxDetectionType.TRAFFIC_CONE,
    "movable_object.barrier": BoxDetectionType.BARRIER,
    # Generic objects
    "movable_object.pushable_pullable": BoxDetectionType.GENERIC_OBJECT,
    "movable_object.debris": BoxDetectionType.GENERIC_OBJECT,
    "static_object.bicycle_rack": BoxDetectionType.GENERIC_OBJECT,
    "animal": BoxDetectionType.GENERIC_OBJECT,
}

NUSCENES_CAMERA_TYPES = {
    PinholeCameraType.CAM_F0: "CAM_FRONT",
    PinholeCameraType.CAM_B0: "CAM_BACK",
    PinholeCameraType.CAM_L0: "CAM_FRONT_LEFT",
    PinholeCameraType.CAM_L1: "CAM_BACK_LEFT",
    PinholeCameraType.CAM_R0: "CAM_FRONT_RIGHT",
    PinholeCameraType.CAM_R1: "CAM_BACK_RIGHT",
}
NUSCENES_DATA_ROOT = Path(os.environ["NUSCENES_DATA_ROOT"])
