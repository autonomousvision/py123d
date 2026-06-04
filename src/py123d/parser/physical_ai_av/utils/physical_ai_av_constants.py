import logging
from typing import Dict, Set

from py123d.datatypes import CameraID
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.sensors.radar import RadarID
from py123d.parser.registry import PhysicalAIAVBoxDetectionLabel

logger = logging.getLogger(__name__)

PHYSICAL_AI_AV_SPLITS: Set[str] = {"physical-ai-av_train", "physical-ai-av_val", "physical-ai-av_test"}

# PHYSICAL_AI_AV_CAMERAS: List[str] = [
#     "camera_front_wide_120fov",
#     "camera_front_tele_30fov",
#     "camera_cross_left_120fov",
#     "camera_cross_right_120fov",
#     "camera_rear_left_70fov",
#     "camera_rear_right_70fov",
#     "camera_rear_tele_30fov",
# ]

PHYSICAL_AI_AV_CAMERA_ID_MAPPING: Dict[str, CameraID] = {
    "camera_front_wide_120fov": CameraID.FTCAM_F0,
    "camera_front_tele_30fov": CameraID.FTCAM_TELE_F0,
    "camera_cross_left_120fov": CameraID.FTCAM_L0,
    "camera_cross_right_120fov": CameraID.FTCAM_R0,
    "camera_rear_left_70fov": CameraID.FTCAM_L1,
    "camera_rear_right_70fov": CameraID.FTCAM_R1,
    "camera_rear_tele_30fov": CameraID.FTCAM_TELE_B0,
}

# PAIAV ships up to nine short-range radars (``radar/<name>/<clip>.<name>.parquet``). The sensor name
# preserves the exact dataset name; the RadarID encodes the spatial position for the unified model.
PHYSICAL_AI_AV_RADAR_ID_MAPPING: Dict[str, RadarID] = {
    "radar_front_center_srr_0": RadarID.RADAR_FRONT,
    "radar_corner_front_left_srr_0": RadarID.RADAR_FRONT_LEFT,
    "radar_corner_front_right_srr_0": RadarID.RADAR_FRONT_RIGHT,
    "radar_side_left_srr_0": RadarID.RADAR_SIDE_LEFT,
    "radar_side_right_srr_0": RadarID.RADAR_SIDE_RIGHT,
    "radar_corner_rear_left_srr_0": RadarID.RADAR_BACK_LEFT,
    "radar_corner_rear_right_srr_0": RadarID.RADAR_BACK_RIGHT,
    "radar_rear_left_srr_0": RadarID.RADAR_REAR_LEFT,
    "radar_rear_right_srr_0": RadarID.RADAR_REAR_RIGHT,
}

# Per-clip ego-vehicle dimensions are read from `calibration/vehicle_dimensions/` at parse
# time — see `physical_ai_av_parser._get_ego_state_metadata`. PAIAV ships at least two
# Hyperion variants with different wheel bases, so a single static metadata instance would
# misrepresent ~⅓ of clips.

PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(
    box_detection_label_class=PhysicalAIAVBoxDetectionLabel,
)

# Mapping from dataset label_class strings to PhysicalAIAVBoxDetectionLabel enum values.
# Keys match the 12 dynamic object classes documented in the NVlabs physical_ai_av wiki
# (https://github.com/NVlabs/physical_ai_av/wiki/5.-Machine-Labels-(Coming-Soon)),
# stored lowercase; ``resolve_physical_ai_av_label`` normalizes input case.
PHYSICAL_AI_AV_LABEL_CLASS_MAPPING: Dict[str, PhysicalAIAVBoxDetectionLabel] = {
    "automobile": PhysicalAIAVBoxDetectionLabel.AUTOMOBILE,
    "heavy_truck": PhysicalAIAVBoxDetectionLabel.HEAVY_TRUCK,
    "bus": PhysicalAIAVBoxDetectionLabel.BUS,
    "train_or_tram_car": PhysicalAIAVBoxDetectionLabel.TRAIN_OR_TRAM_CAR,
    "trolley_bus": PhysicalAIAVBoxDetectionLabel.TROLLEY_BUS,
    "other_vehicle": PhysicalAIAVBoxDetectionLabel.OTHER_VEHICLE,
    "trailer": PhysicalAIAVBoxDetectionLabel.TRAILER,
    "person": PhysicalAIAVBoxDetectionLabel.PERSON,
    "stroller": PhysicalAIAVBoxDetectionLabel.STROLLER,
    "rider": PhysicalAIAVBoxDetectionLabel.RIDER,
    "animal": PhysicalAIAVBoxDetectionLabel.ANIMAL,
    "protruding_object": PhysicalAIAVBoxDetectionLabel.PROTRUDING_OBJECT,
}

_UNKNOWN_PHYSICAL_AI_AV_LABELS_WARNED: set = set()


def resolve_physical_ai_av_label(label_str: str) -> PhysicalAIAVBoxDetectionLabel:
    """Resolve a dataset ``label_class`` string to a :class:`PhysicalAIAVBoxDetectionLabel`.

    Falls back to ``OTHER_VEHICLE`` for unknown labels and logs a warning the first
    time each unrecognized label is seen so silent miscategorization is surfaced.
    """
    key = label_str.lower() if isinstance(label_str, str) else label_str
    label = PHYSICAL_AI_AV_LABEL_CLASS_MAPPING.get(key)
    if label is None:
        if label_str not in _UNKNOWN_PHYSICAL_AI_AV_LABELS_WARNED:
            _UNKNOWN_PHYSICAL_AI_AV_LABELS_WARNED.add(label_str)
            logger.warning(
                "Unknown Physical AI AV label_class %r; falling back to OTHER_VEHICLE. Expected one of: %s",
                label_str,
                sorted(PHYSICAL_AI_AV_LABEL_CLASS_MAPPING.keys()),
            )
        label = PhysicalAIAVBoxDetectionLabel.OTHER_VEHICLE
    return label
