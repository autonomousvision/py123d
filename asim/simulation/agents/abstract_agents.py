from abc import abstractmethod
from typing import List, Optional

from asim.common.datatypes.detection.detection import BoxDetection
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.scene.abstract_scene import AbstractScene


class AbstractAgents:

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    @abstractmethod
    def reset(
        self,
        map_api: AbstractMap,
        target_agents: List[BoxDetection],
        non_target_agents: List[BoxDetection],
        scene: Optional[AbstractScene] = None,
    ) -> List[BoxDetection]:
        raise NotImplementedError

    @abstractmethod
    def step(self, non_target_agents: List[BoxDetection]) -> List[BoxDetection]:
        raise NotImplementedError
