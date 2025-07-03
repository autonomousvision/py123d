from dataclasses import dataclass

from asim.common.datatypes.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from asim.common.datatypes.recording.abstract_recording import Recording


@dataclass
class DetectionRecording(Recording):

    box_detections: BoxDetectionWrapper
    traffic_light_detections: TrafficLightDetectionWrapper
