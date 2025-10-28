from dataclasses import dataclass
from typing import List, Optional

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.time.time_point import TimePoint


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

    lane_id: int
    status: TrafficLightStatus
    timepoint: Optional[TimePoint] = None


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
