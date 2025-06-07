from dataclasses import dataclass

from asim.dataset.observation.abstract_observation import Observation
from asim.dataset.observation.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper


@dataclass
class DetectionObservation(Observation):

    box_detections: BoxDetectionWrapper
    traffic_light_detections: TrafficLightDetectionWrapper
