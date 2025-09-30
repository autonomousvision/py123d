from typing import List, Optional, Tuple, Type

from d123.common.datatypes.detection.detection import BoxDetection
from d123.common.datatypes.detection.detection_types import DetectionType
from d123.common.datatypes.recording.abstract_recording import Recording
from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.datatypes.scene.arrow.utils.conversion import BoxDetectionWrapper
from d123.simulation.agents.abstract_agents import AbstractAgents

# from d123.simulation.agents.path_following import PathFollowingAgents
from d123.simulation.agents.idm_agents import IDMAgents

# from d123.simulation.agents.smart_agents import SMARTAgents
from d123.simulation.observation.abstract_observation import AbstractObservation
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


class AgentsObservation(AbstractObservation):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    def __init__(self, agents: AbstractAgents) -> None:
        super().__init__()
        self._scene: Optional[AbstractScene] = None
        # self._agents: AbstractAgents = ConstantVelocityAgents()
        self._agents: AbstractAgents = IDMAgents()
        # self._agents: AbstractAgents = SMARTAgents()

    def recording_type(self) -> Type[Recording]:
        return DetectionRecording

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

    def step(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        current_ego_state: EgoStateSE2,
    ) -> DetectionRecording:
        assert self._scene is not None, "Scene must be provided for log replay observation."
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
