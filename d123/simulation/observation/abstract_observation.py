import abc
from abc import abstractmethod
from typing import Optional, Type

from d123.common.datatypes.recording.abstract_recording import Recording
from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


class AbstractObservation(abc.ABC):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    @abstractmethod
    def recording_type(self) -> Type[Recording]:
        pass

    @abstractmethod
    def reset(self, scene: Optional[AbstractScene]) -> DetectionRecording:
        pass

    @abstractmethod
    def step(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        current_ego_state: EgoStateSE2,
    ) -> DetectionRecording:
        pass
