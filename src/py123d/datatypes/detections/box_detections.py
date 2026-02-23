from __future__ import annotations

from functools import cached_property
from typing import List, Optional, Union

import shapely

from py123d.conversion.registry import BoxDetectionLabel, DefaultBoxDetectionLabel
from py123d.datatypes.time import Timestamp
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, OccupancyMap2D, PoseSE2, PoseSE3, Vector2D, Vector3D


class BoxDetectionMetadata:
    """Stores data about the box detection, including its label, track token, number of Lidar points, and timestamp."""

    __slots__ = ("_label", "_track_token", "_num_lidar_points", "_timestamp")

    def __init__(
        self,
        label: BoxDetectionLabel,
        track_token: str,
        num_lidar_points: Optional[int] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> None:
        """Initialize a BoxDetectionMetadata instance.

        :param label: The label of the detection.
        :param track_token: The track token of the detection.
        :param num_lidar_points: The number of Lidar points, defaults to None.
        :param timestamp: The timestamp of the detection, defaults to None.
        """
        self._label = label
        self._track_token = track_token
        self._num_lidar_points = num_lidar_points
        self._timestamp = timestamp

    @property
    def label(self) -> BoxDetectionLabel:
        """The :class:`~py123d.datatypes.detections.BoxDetectionLabel`, from the original dataset's label set."""
        return self._label

    @property
    def track_token(self) -> str:
        """The unique track token of the detection, consistent across frames."""
        return self._track_token

    @property
    def num_lidar_points(self) -> Optional[int]:
        """Optionally, the number of Lidar points associated with the detection."""
        return self._num_lidar_points

    @property
    def timestamp(self) -> Optional[Timestamp]:
        """Optionally, the :class:`~py123d.datatypes.time.Timestamp` of the detection."""
        return self._timestamp

    @property
    def default_label(self) -> DefaultBoxDetectionLabel:
        """The unified :class:`~py123d.conversion.registry.DefaultBoxDetectionLabel`
        corresponding to the detection's label.
        """
        return self.label.to_default()


class BoxDetectionSE2:
    """Detected, tracked, and oriented bounding box 2D space."""

    __slots__ = ("_metadata", "_bounding_box_se2", "_velocity_2d")

    def __init__(
        self,
        metadata: BoxDetectionMetadata,
        bounding_box_se2: BoundingBoxSE2,
        velocity_2d: Optional[Vector2D] = None,
    ) -> None:
        """Initialize a BoxDetectionSE2 instance.

        :param metadata: The :class:`BoxDetectionMetadata` of the detection.
        :param bounding_box_se2: The :class:`~py123d.datatypes.geometry.BoundingBoxSE2` of the detection.
        :param velocity: Optionally, a :class:`~py123d.geometry.Vector2D` representing the velocity.
        """

        self._metadata = metadata
        self._bounding_box_se2 = bounding_box_se2
        self._velocity_2d = velocity_2d

    @property
    def metadata(self) -> BoxDetectionMetadata:
        """The :class:`BoxDetectionMetadata` of the detection."""
        return self._metadata

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`~py123d.geometry.BoundingBoxSE2` of the detection."""
        return self._bounding_box_se2

    @property
    def velocity_2d(self) -> Optional[Vector2D]:
        """The :class:`~py123d.geometry.Vector2D` representing the velocity."""
        return self._velocity_2d

    @property
    def center_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` representing the center of the bounding box."""
        return self.bounding_box_se2.center_se2

    @property
    def shapely_polygon(self) -> shapely.geometry.Polygon:
        """The shapely polygon of the bounding box in 2D space."""
        return self.bounding_box_se2.shapely_polygon

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        """Returns self to maintain interface consistency."""
        return self


class BoxDetectionSE3:
    """Detected, tracked, and oriented bounding box 3D space."""

    __slots__ = ("_metadata", "_bounding_box_se3", "_velocity")

    def __init__(
        self,
        metadata: BoxDetectionMetadata,
        bounding_box_se3: BoundingBoxSE3,
        velocity_3d: Optional[Vector3D] = None,
    ) -> None:
        """Initialize a BoxDetectionSE3 instance.

        :param metadata: The :class:`BoxDetectionMetadata` of the detection.
        :param bounding_box_se3: The :class:`~py123d.datatypes.geometry.BoundingBoxSE3` of the detection.
        :param velocity_3d: Optionally, a :class:`~py123d.geometry.Vector3D` representing the velocity.
        """
        self._metadata = metadata
        self._bounding_box_se3 = bounding_box_se3
        self._velocity = velocity_3d

    @property
    def metadata(self) -> BoxDetectionMetadata:
        """The :class:`BoxDetectionMetadata` of the detection."""
        return self._metadata

    @property
    def bounding_box_se3(self) -> BoundingBoxSE3:
        """The :class:`~py123d.geometry.BoundingBoxSE3` of the detection."""
        return self._bounding_box_se3

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The SE2 projection :class:`~py123d.geometry.BoundingBoxSE2` of the SE3 bounding box."""
        return self.bounding_box_se3.bounding_box_se2

    @property
    def velocity_3d(self) -> Optional[Vector3D]:
        """The :class:`~py123d.geometry.Vector3D` representing the velocity."""
        return self._velocity

    @property
    def velocity_2d(self) -> Optional[Vector2D]:
        """The 2D projection :class:`~py123d.geometry.Vector2D` of the 3D velocity."""
        return Vector2D(self._velocity.x, self._velocity.y) if self._velocity else None

    @property
    def center_se3(self) -> PoseSE3:
        """The :class:`~py123d.geometry.PoseSE3` representing the center of the bounding box."""
        return self.bounding_box_se3.center_se3

    @property
    def center_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` representing the center of the SE2 bounding box."""
        return self.bounding_box_se2.center_se2

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE2` projection of this SE3 box detection."""
        return BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=Vector2D(self.velocity_3d.x, self.velocity_3d.y) if self.velocity_3d else None,
        )

    @property
    def shapely_polygon(self) -> shapely.geometry.Polygon:
        """The shapely polygon of the bounding box in 2D space."""
        return self.bounding_box_se3.shapely_polygon


BoxDetection = Union[BoxDetectionSE2, BoxDetectionSE3]


class BoxDetectionWrapper:
    """The BoxDetectionWrapper is a container for multiple box detections.
    It provides methods to access individual detections as well as to retrieve a detection by track token.
    The wrapper is used to read and write box detections from/to logs.
    """

    def __init__(self, box_detections: List[BoxDetection]) -> None:
        """Initialize a BoxDetectionWrapper instance.

        :param box_detections: A list of :class:`BoxDetection` instances.
        """
        self._box_detections = box_detections

    @property
    def box_detections(self) -> List[BoxDetection]:
        """List of individual :class:`BoxDetectionSE2` or :class:`BoxDetectionSE3`."""
        return self._box_detections

    def __getitem__(self, index: int) -> BoxDetection:
        """Retrieve a box detection by its index.

        :param index: The index of the box detection.
        :return: The box detection at the given index.
        """
        return self._box_detections[index]

    def __len__(self) -> int:
        """Number of box detections."""
        return len(self._box_detections)

    def __iter__(self):
        """Iterator over box detections."""
        return iter(self._box_detections)

    def get_detection_by_track_token(self, track_token: str) -> Optional[Union[BoxDetectionSE2, BoxDetectionSE3]]:
        """Retrieve a box detection by its track token.

        :param track_token: The track token of the box detection.
        :return: The box detection with the given track token, or None if not found.
        """

        box_detection: Optional[BoxDetection] = None
        for detection in self.box_detections:
            if detection.metadata.track_token == track_token:
                box_detection = detection
                break
        return box_detection

    @cached_property
    def occupancy_map_2d(self) -> OccupancyMap2D:
        """The :class:`~py123d.geometry.OccupancyMap2D` representing the 2D occupancy of all box detections."""
        ids = [detection.metadata.track_token for detection in self.box_detections]
        geometries = [detection.shapely_polygon for detection in self.box_detections]
        return OccupancyMap2D(geometries=geometries, ids=ids)
