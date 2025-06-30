from typing import List, Optional, Tuple

from asim.dataset.arrow.conversion import BoxDetectionWrapper
from asim.dataset.recording.detection.detection import BoxDetection
from asim.dataset.recording.detection.detection_types import DetectionType
from asim.dataset.recording.detection_recording import DetectionRecording
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.agents.abstract_agents import AbstractAgents
from asim.simulation.agents.constant_velocity_agents import ConstantVelocityAgents

# from asim.simulation.agents.path_following import PathFollowingAgents
# from asim.simulation.agents.idm_agents import IDMAgents
# from asim.simulation.agents.smart_agents import SMARTAgents
from asim.simulation.observation.abstract_observation import AbstractObservation


class AgentsObservation(AbstractObservation):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    def __init__(self, agents: AbstractAgents) -> None:
        super().__init__()
        self._scene: Optional[AbstractScene] = None
        self._agents: AbstractAgents = ConstantVelocityAgents()
        # self._agents: AbstractAgents = IDMAgents()
        # self._agents: AbstractAgents = SMARTAgents()

    def initialize(self) -> None:
        pass

    def reset(self, scene: Optional[AbstractScene]) -> DetectionRecording:
        assert scene is not None, "Scene must be provided for log replay observation."
        self._scene = scene
        self._iteration = 0

        cars, non_cars, _ = _filter_agents_by_type(
            self._scene.get_box_detections_at_iteration(self._iteration),
            detection_types=[DetectionType.VEHICLE],
        )
        cars = self._agents.reset(
            map_api=self._scene.map_api,
            target_agents=cars,
            non_target_agents=non_cars,
            scene=self._scene if self._agents.requires_scene else None,
        )
        return DetectionRecording(
            box_detections=BoxDetectionWrapper(cars + non_cars),
            traffic_light_detections=self._scene.get_traffic_light_detections_at_iteration(self._iteration),
        )

    def step(self) -> DetectionRecording:
        self._iteration += 1
        _, non_cars, _ = _filter_agents_by_type(
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
) -> Tuple[List[BoxDetection], List[BoxDetection], List[int]]:

    in_types, not_in_types, in_indices = [], [], []
    for detection_idx, detection in enumerate(detections):
        if detection.metadata.detection_type in detection_types:
            in_types.append(detection)
            in_indices.append(detection_idx)
        else:
            not_in_types.append(detection)

    return in_types, not_in_types, in_indices
