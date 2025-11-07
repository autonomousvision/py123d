from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Union

import shapely

from py123d.conversion.registry.box_detection_label_registry import BoxDetectionLabel, DefaultBoxDetectionLabel
from py123d.datatypes.time.time_point import TimePoint
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, OccupancyMap2D, StateSE2, StateSE3, Vector2D, Vector3D


@dataclass
class BoxDetectionMetadata:
    """Store metadata for a detected bounding box.

    Examples
    --------

    .. code-block:: python

        from mymodule import my_function

        result = my_function()
        print(result)

    """

    label: BoxDetectionLabel
    track_token: str
    num_lidar_points: Optional[int] = None
    timepoint: Optional[TimePoint] = None

    @property
    def default_label(self) -> DefaultBoxDetectionLabel:
        """The default label of the detection.

        :return: The default label.
        """
        return self.label.to_default()


@dataclass
class BoxDetectionSE2:
    """Store a 2D bounding box detection.

    Example:
        >>> from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
        >>> from py123d.datatypes.detections import BoxDetectionMetadata, BoxDetectionSE2
        >>> from py123d.geometry import BoundingBoxSE2, StateSE2, Vector2D
        >>> metadata = BoxDetectionMetadata(
        ...     label=DefaultBoxDetectionLabel.VEHICLE,
        ...     track_token="track_123",
        ... )
        >>> bounding_box = BoundingBoxSE2(
        ...     center=StateSE2(x=0.0, y=0.0, yaw=0.0),
        ...     length=4.0,
        ...     width=2.0,
        ... )
        >>> detection = BoxDetectionSE2(
        ...     metadata=metadata,
        ...     bounding_box_se2=bounding_box,
        ...     velocity=Vector2D(x=1.0, y=0.0),
        ... )
    """

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

    def get_detection_by_track_token(self, track_token: str) -> Optional[BoxDetection]:
        box_detection: Optional[BoxDetection] = None
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
