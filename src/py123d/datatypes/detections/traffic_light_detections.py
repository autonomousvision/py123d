from typing import List, Optional

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.time import Timestamp


class TrafficLightStatus(SerialIntEnum):
    """
    Enum for that represents the status of a traffic light.
    """

    GREEN = 0
    """Green light is on."""

    YELLOW = 1
    """Yellow light is on."""

    RED = 2
    """Red light is on."""

    OFF = 3
    """Traffic light is off."""

    UNKNOWN = 4
    """Traffic light status is unknown."""


class TrafficLightDetection:
    """
    Single traffic light detection if a lane, that includes the lane id, status (green, yellow, red, off, unknown),
    and optional timestamp of the detection.
    """

    __slots__ = ("_lane_id", "_status", "_timestamp")

    def __init__(self, lane_id: int, status: TrafficLightStatus, timestamp: Optional[Timestamp] = None) -> None:
        """Initialize a TrafficLightDetection instance.

        :param lane_id: The lane id associated with the traffic light detection.
        :param status: The status of the traffic light (green, yellow, red, off, unknown).
        :param timestamp: The optional timestamp of the detection.
        """

        self._lane_id = lane_id
        self._status = status
        self._timestamp = timestamp

    @property
    def lane_id(self) -> int:
        """The lane id associated with the traffic light detection."""
        return self._lane_id

    @property
    def status(self) -> TrafficLightStatus:
        """The :class:`TrafficLightStatus` of the traffic light detection."""
        return self._status

    @property
    def timestamp(self) -> Optional[Timestamp]:
        """The optional :class:`~py123d.datatypes.time.TimePoint` of the traffic light detection."""
        return self._timestamp


class TrafficLightDetectionWrapper:
    """The TrafficLightDetectionWrapper is a container for multiple traffic light detections.
    It provides methods to access individual detections as well as to retrieve a detection by lane id.
    The wrapper is is used in to read and write traffic light detections from/to logs.
    """

    __slots__ = ("_traffic_light_detections",)

    def __init__(self, traffic_light_detections: List[TrafficLightDetection]) -> None:
        """Initialize a TrafficLightDetectionWrapper instance.

        :param traffic_light_detections: List of :class:`TrafficLightDetection`.
        """
        self._traffic_light_detections = traffic_light_detections

    @property
    def traffic_light_detections(self) -> List[TrafficLightDetection]:
        """List of individual :class:`TrafficLightDetection`."""
        return self._traffic_light_detections

    def __getitem__(self, index: int) -> TrafficLightDetection:
        """Retrieve a traffic light detection by its index.

        :param index: The index of the traffic light detection.
        :return: :class:`TrafficLightDetection` at the given index.
        """
        return self.traffic_light_detections[index]

    def __len__(self) -> int:
        """The number of traffic light detections in the wrapper."""
        return len(self.traffic_light_detections)

    def __iter__(self):
        """Iterator over the traffic light detections in the wrapper."""
        return iter(self.traffic_light_detections)

    def get_detection_by_lane_id(self, lane_id: int) -> Optional[TrafficLightDetection]:
        """Retrieve a traffic light detection by its lane id.

        :param lane_id: The lane id to search for.
        :return: The traffic light detection for the given lane id, or None if not found.
        """
        traffic_light_detection: Optional[TrafficLightDetection] = None
        for detection in self.traffic_light_detections:
            if int(detection.lane_id) == int(lane_id):
                traffic_light_detection = detection
                break
        return traffic_light_detection
