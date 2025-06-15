from typing import List, Optional, Tuple

from asim.dataset.arrow.conversion import BoxDetectionWrapper
from asim.dataset.recording.detection.detection import BoxDetection
from asim.dataset.recording.detection.detection_types import DetectionType
from asim.dataset.recording.detection_recording import DetectionRecording
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.agents.abstract_agents import AbstractAgents
from asim.simulation.agents.constant_velocity_agents import ConstantVelocityAgents
from asim.simulation.observation.abstract_observation import AbstractObservation


class AgentsObservation(AbstractObservation):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    def __init__(self, agents: AbstractAgents) -> None:
        super().__init__()
        self._scene: Optional[AbstractScene] = None
        self._agents: AbstractAgents = ConstantVelocityAgents()

    def initialize(self) -> None:
        pass

    def reset(self, scene: Optional[AbstractScene]) -> DetectionRecording:
        assert scene is not None, "Scene must be provided for log replay observation."
        self._scene = scene
        self._iteration = 0

        cars, non_cars = _filter_agents_by_type(
            self._scene.get_box_detections_at_iteration(self._iteration),
            detection_types=[DetectionType.VEHICLE],
        )
        cars = self._agents.reset(
            map_api=self._scene.map_api,
            target_agents=cars,
            non_target_agents=non_cars,
        )
        return DetectionRecording(
            box_detections=BoxDetectionWrapper(cars + non_cars),
            traffic_light_detections=self._scene.get_traffic_light_detections_at_iteration(self._iteration),
        )

    def step(self) -> DetectionRecording:
        self._iteration += 1
        _, non_cars = _filter_agents_by_type(
            self._scene.get_box_detections_at_iteration(self._iteration),
            detection_types=[DetectionType.VEHICLE],
        )
        cars = self._agents.step(non_target_agents=non_cars)
        return DetectionRecording(
            box_detections=BoxDetectionWrapper(cars + non_cars),
            traffic_light_detections=self._scene.get_traffic_light_detections_at_iteration(self._iteration),
        )


def _filter_agents_by_type(
    detections: BoxDetectionWrapper, detection_types: List[DetectionType]
) -> Tuple[List[BoxDetection], List[BoxDetection]]:
    """
    Filter agents by detection type.
    :param detections: The detection recording to filter.
    :param detection_type: The detection type to filter by.
    :return: A new DetectionRecording with only the specified detection type.
    """
    in_types, not_in_types = [], []
    for detection in detections:
        if detection.metadata.detection_type in detection_types:
            in_types.append(detection)
        else:
            not_in_types.append(detection)

    return in_types, not_in_types
