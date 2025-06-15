from abc import abstractmethod
from typing import List

from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.recording.detection.detection import BoxDetection


class AbstractAgents:

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    @abstractmethod
    def reset(
        self, map_api: AbstractMap, target_agents: List[BoxDetection], non_target_agents: List[BoxDetection]
    ) -> List[BoxDetection]:
        pass

    def step(self, non_target_agents: List[BoxDetection]):
        pass
