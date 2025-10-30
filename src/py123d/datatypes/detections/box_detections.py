from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, List, Optional, Union

import shapely

from py123d.datatypes.detections.box_detection_types import BoxDetectionType
from py123d.datatypes.time.time_point import TimePoint
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, OccupancyMap2D, StateSE2, StateSE3, Vector2D, Vector3D


@dataclass
class BoxDetectionMetadata:

    box_detection_type: BoxDetectionType
    track_token: str
    confidence: Optional[float] = None  # Confidence score of the detection, if available
    num_lidar_points: Optional[int] = None  # Number of LiDAR points within the bounding box
    timepoint: Optional[TimePoint] = None  # TimePoint when the detection was made, if available

    @property
    def default_box_detection_type(self) -> BoxDetectionType:
        return self.box_detection_type.to_default_type()


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

    def get_box_detections_by_types(self, detection_types: Iterable[BoxDetectionType]) -> List[BoxDetection]:
        return [
            detection for detection in self.box_detections if detection.metadata.box_detection_type in detection_types
        ]

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
