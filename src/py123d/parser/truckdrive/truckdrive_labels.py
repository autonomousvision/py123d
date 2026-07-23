"""TruckDrive box detection label registry and decoding helpers."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from py123d.datatypes.detections.box_detection_label import (
    BoxDetectionLabel,
    DefaultBoxDetectionLabel,
    register_box_detection_label,
)
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.parser.truckdrive.truckdrive_constants import LABEL_MAPPING, MAPPED_CATEGORIES

logger = logging.getLogger(__name__)

_MAPPED_CATEGORY_ID_TO_ENUM: Dict[int, "TruckDriveBoxDetectionLabel"] = {}


@register_box_detection_label
class TruckDriveBoxDetectionLabel(BoxDetectionLabel):
    """TruckDrive mapped detection categories."""

    BIKE = 0
    PASSENGER_CAR = 1
    PERSON = 2
    ROAD_OBSTRUCTION = 3
    SEMI_TRUCK_CAB = 4
    SEMI_TRUCK_TRAILER = 5
    VEHICLE = 6
    TRAFFIC_SIGN = 7
    EMERGENCY_VEHICLE = 8

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            TruckDriveBoxDetectionLabel.BIKE: DefaultBoxDetectionLabel.TWO_WHEELER,
            TruckDriveBoxDetectionLabel.PASSENGER_CAR: DefaultBoxDetectionLabel.VEHICLE,
            TruckDriveBoxDetectionLabel.PERSON: DefaultBoxDetectionLabel.PERSON,
            TruckDriveBoxDetectionLabel.ROAD_OBSTRUCTION: DefaultBoxDetectionLabel.BARRIER,
            TruckDriveBoxDetectionLabel.SEMI_TRUCK_CAB: DefaultBoxDetectionLabel.VEHICLE,
            TruckDriveBoxDetectionLabel.SEMI_TRUCK_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            TruckDriveBoxDetectionLabel.VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            TruckDriveBoxDetectionLabel.TRAFFIC_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            TruckDriveBoxDetectionLabel.EMERGENCY_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
        }
        return mapping[self]


def _build_mapped_category_lookup() -> Dict[int, TruckDriveBoxDetectionLabel]:
    name_to_member = {
        "Bike": TruckDriveBoxDetectionLabel.BIKE,
        "Passenger-Car": TruckDriveBoxDetectionLabel.PASSENGER_CAR,
        "Person": TruckDriveBoxDetectionLabel.PERSON,
        "RoadObstruction": TruckDriveBoxDetectionLabel.ROAD_OBSTRUCTION,
        "SemiTruck-Cab": TruckDriveBoxDetectionLabel.SEMI_TRUCK_CAB,
        "SemiTruck-Trailer": TruckDriveBoxDetectionLabel.SEMI_TRUCK_TRAILER,
        "Vehicle": TruckDriveBoxDetectionLabel.VEHICLE,
        "TrafficSign": TruckDriveBoxDetectionLabel.TRAFFIC_SIGN,
        "EmergencyVehicle": TruckDriveBoxDetectionLabel.EMERGENCY_VEHICLE,
    }
    lookup: Dict[int, TruckDriveBoxDetectionLabel] = {}
    for category_name, category_id in MAPPED_CATEGORIES.items():
        if category_id == -1:
            continue
        member = name_to_member.get(category_name)
        if member is None:
            raise ValueError(f"Unknown mapped category name in TruckDrive metainfo: {category_name}")
        lookup[category_id] = member
    return lookup


_MAPPED_CATEGORY_ID_TO_ENUM = _build_mapped_category_lookup()

TRUCKDRIVE_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=TruckDriveBoxDetectionLabel)


def decode_class_id(class_id: str) -> Optional[TruckDriveBoxDetectionLabel]:
    """Decode a TruckDrive ``class-id`` string into a mapped label enum member.

    :param class_id: Raw class-id string from the annotation JSON.
    :return: Mapped label enum member, or ``None`` for DontCare / unknown labels.
    """
    if class_id not in LABEL_MAPPING:
        logger.warning("Unknown TruckDrive class-id: %s", class_id)
        return None

    mapped_id = LABEL_MAPPING[class_id]
    if mapped_id == -1:
        return None

    label = _MAPPED_CATEGORY_ID_TO_ENUM.get(mapped_id)
    if label is None:
        logger.warning("Unknown TruckDrive mapped category id %s for class-id %s", mapped_id, class_id)
    return label
