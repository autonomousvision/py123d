from abc import abstractmethod
from typing import Optional

from asim.common.datatypes.recording.detection_recording import DetectionRecording
from asim.dataset.scene.abstract_scene import AbstractScene


class AbstractObservation:

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize observation if needed.
        """

    @abstractmethod
    def reset(self, scene: Optional[AbstractScene]) -> DetectionRecording:
        pass

    @abstractmethod
    def step(self) -> DetectionRecording:
        pass
