from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, List, Optional, Union

import shapely

from d123.common.utils.enums import SerialIntEnum
from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.time.time_point import TimePoint
from d123.geometry import BoundingBoxSE2, BoundingBoxSE3, OccupancyMap2D, StateSE2, StateSE3, Vector2D, Vector3D


@dataclass
class BoxDetectionMetadata:

    detection_type: DetectionType
    track_token: str
    timepoint: Optional[TimePoint] = None  # TimePoint when the detection was made, if available
    confidence: Optional[float] = None  # Confidence score of the detection, if available
    num_lidar_points: Optional[int] = None  # Number of LiDAR points within the bounding box


@dataclass
class BoxDetectionSE2:

    metadata: BoxDetectionMetadata
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

    metadata: BoxDetectionMetadata
    bounding_box_se3: BoundingBoxSE3
    velocity: Optional[Vector3D] = None

    @property
    def shapely_polygon(self) -> shapely.geometry.Polygon:
        return self.bounding_box_se3.shapely_polygon

    @property
    def center(self) -> StateSE3:
        return self.bounding_box_se3.center

    @property
    def center_se3(self) -> StateSE3:
        return self.bounding_box_se3.center_se3

    @property
    def bounding_box(self) -> BoundingBoxSE3:
        return self.bounding_box_se3

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        return self.bounding_box_se3.bounding_box_se2

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        return BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity=Vector2D(self.velocity.x, self.velocity.y) if self.velocity else None,
        )


BoxDetection = Union[BoxDetectionSE2, BoxDetectionSE3]


@dataclass
class BoxDetectionWrapper:

    box_detections: List[BoxDetection]

    def __getitem__(self, index: int) -> BoxDetection:
        return self.box_detections[index]

    def __len__(self) -> int:
        return len(self.box_detections)

    def __iter__(self):
        return iter(self.box_detections)

    def get_box_detections_by_types(self, detection_types: Iterable[DetectionType]) -> List[BoxDetection]:
        return [detection for detection in self.box_detections if detection.metadata.detection_type in detection_types]

    def get_detection_by_track_token(self, track_token: str) -> BoxDetection | None:
        box_detection: BoxDetection | None = None
        for detection in self.box_detections:
            if detection.metadata.track_token == track_token:
                box_detection = detection
                break
        return box_detection

    @cached_property
    def occupancy_map(self) -> OccupancyMap2D:
        ids = [detection.metadata.track_token for detection in self.box_detections]
        geometries = [detection.bounding_box.shapely_polygon for detection in self.box_detections]
        return OccupancyMap2D(geometries=geometries, ids=ids)


class TrafficLightStatus(SerialIntEnum):
    """
    Enum for TrafficLightStatus.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    OFF = 3
    UNKNOWN = 4


@dataclass
class TrafficLightDetection:

    timepoint: TimePoint  # TODO: Consider removing or making optional
    lane_id: int
    status: TrafficLightStatus


@dataclass
class TrafficLightDetectionWrapper:

    traffic_light_detections: List[TrafficLightDetection]

    def __getitem__(self, index: int) -> TrafficLightDetection:
        return self.traffic_light_detections[index]

    def __len__(self) -> int:
        return len(self.traffic_light_detections)

    def __iter__(self):
        return iter(self.traffic_light_detections)

    def get_detection_by_lane_id(self, lane_id: int) -> Optional[TrafficLightDetection]:
        traffic_light_detection: Optional[TrafficLightDetection] = None
        for detection in self.traffic_light_detections:
            if int(detection.lane_id) == int(lane_id):
                traffic_light_detection = detection
                break
        return traffic_light_detection


@dataclass
class DetectionRecording:

    box_detections: BoxDetectionWrapper
    traffic_light_detections: TrafficLightDetectionWrapper
