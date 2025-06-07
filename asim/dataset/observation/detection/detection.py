# class


from dataclasses import dataclass
from typing import List, Optional, Union

import shapely

from asim.common.geometry.base import StateSE2, StateSE3
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE2, BoundingBoxSE3
from asim.common.geometry.vector import Vector2D, Vector3D
from asim.common.time.time_point import TimePoint
from asim.common.utils.enums import SerialIntEnum
from asim.dataset.observation.detection.detection_types import DetectionType

# from collections.abc import Iterable, Sequence


@dataclass
class DetectionMetadata:

    detection_type: DetectionType
    timepoint: TimePoint
    track_token: str
    confidence: Optional[float] = None


@dataclass
class BoxDetectionSE2:

    metadata: DetectionMetadata
    bounding_box_se2: BoundingBoxSE2
    velocity: Optional[Vector2D] = None

    @property
    def shapely_polygon(self) -> shapely.geometry.Polygon:
        return self.bounding_box_se2.shapely_polygon

    @property
    def center(self) -> StateSE2:
        return self.bounding_box_se2.center

    @property
    def bounding_box(self) -> BoundingBoxSE2:
        return self.bounding_box_se2


@dataclass
class BoxDetectionSE3:

    metadata: DetectionMetadata
    bounding_box_se3: BoundingBoxSE3
    velocity: Optional[Vector3D] = None

    @property
    def shapely_polygon(self) -> shapely.geometry.Polygon:
        return self.bounding_box_se3.shapely_polygon

    @property
    def center(self) -> StateSE3:
        return self.bounding_box_se3.center

    @property
    def bounding_box(self) -> BoundingBoxSE3:
        return self.bounding_box_se3


BoxDetection = Union[BoxDetectionSE2, BoxDetectionSE3]


@dataclass
class BoxDetectionWrapper:
    # TODO:
    # - Add occupancy map property

    box_detections: List[BoxDetection]

    def __getitem__(self, index):
        return self.box_detections[index]

    def __len__(self):
        return len(self.box_detections)

    def __iter__(self):
        return iter(self.box_detections)


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

    def __getitem__(self, index):
        return self.traffic_light_detections[index]

    def __len__(self):
        return len(self.traffic_light_detections)

    def __iter__(self):
        return iter(self.traffic_light_detections)
