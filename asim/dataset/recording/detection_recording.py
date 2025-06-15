from dataclasses import dataclass

from asim.dataset.recording.abstract_recording import Recording
from asim.dataset.recording.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper


@dataclass
class DetectionRecording(Recording):

    box_detections: BoxDetectionWrapper
    traffic_light_detections: TrafficLightDetectionWrapper
