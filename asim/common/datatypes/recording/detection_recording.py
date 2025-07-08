from dataclasses import dataclass

from asim.common.datatypes.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from asim.common.datatypes.recording.abstract_recording import Recording

# TODO: Reconsider if these "wrapper" datatypes are necessary.
# Might be needed to package multiple datatypes into a single object (e.g. as planner input)
# On the other hand, an enum based dictionary might be more flexible.


@dataclass
class DetectionRecording(Recording):

    box_detections: BoxDetectionWrapper
    traffic_light_detections: TrafficLightDetectionWrapper
