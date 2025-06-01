# class


from dataclasses import dataclass
from typing import List, Optional, Union

from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE2, BoundingBoxSE3
from asim.common.geometry.vector import Vector2D, Vector3D
from asim.common.time.time_point import TimePoint
from asim.common.utils.enums import SerialIntEnum
from asim.dataset.observation.detection.detection_types import DetectionType


@dataclass
class DetectionMetdata:

    detection_type: DetectionType
    timepoint: TimePoint
    track_token: str
    confidence: Optional[float] = None


@dataclass
class BoxDetectionSE2:

    metadata: DetectionMetdata
    bounding_box_se3: BoundingBoxSE2
    velocity: Optional[Vector2D] = None


@dataclass
class BoxDetectionSE3:

    metadata: DetectionMetdata
    bounding_box_se3: BoundingBoxSE3
    velocity: Optional[Vector3D] = None


BoxDetection = Union[BoxDetectionSE2, BoxDetectionSE3]


@dataclass
class BoxDetectionWrapper:
    box_detections: List[BoxDetection]


class TrafficLightStatus(SerialIntEnum):
    """
    Enum for TrafficLightStatus.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3


@dataclass
class TrafficLightDetection:

    timepoint: TimePoint
    lane_id: int
    status: TrafficLightStatus


@dataclass
class TrafficLightDetectionWrapper:
    traffic_light_detections: List[TrafficLightDetection]
