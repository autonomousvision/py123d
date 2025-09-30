from typing import Optional, Type

from d123.common.datatypes.recording.abstract_recording import Recording
from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.simulation.observation.abstract_observation import AbstractObservation
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


class LogReplayObservation(AbstractObservation):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    def __init__(self) -> None:
        """
        Initialize the log replay observation.
        """
        super().__init__()
        self._scene: Optional[AbstractScene] = None

    def recording_type(self) -> Type[Recording]:
        return DetectionRecording

    def reset(self, scene: Optional[AbstractScene]) -> DetectionRecording:
        assert scene is not None, "Scene must be provided for log replay observation."
        self._scene = scene
        self._iteration = 0

        return DetectionRecording(
            box_detections=self._scene.get_box_detections_at_iteration(self._iteration),
            traffic_light_detections=self._scene.get_traffic_light_detections_at_iteration(self._iteration),
        )

    def step(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        current_ego_state: EgoStateSE2,
    ) -> DetectionRecording:
        self._iteration += 1
        return DetectionRecording(
            box_detections=self._scene.get_box_detections_at_iteration(self._iteration),
            traffic_light_detections=self._scene.get_traffic_light_detections_at_iteration(self._iteration),
        )
