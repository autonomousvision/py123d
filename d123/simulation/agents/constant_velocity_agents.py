import copy
from abc import abstractmethod
from typing import List, Optional

from d123.common.datatypes.detection.detection import BoxDetection, BoxDetectionSE2
from d123.conversion.maps.abstract_map import AbstractMap
from d123.conversion.scene.abstract_scene import AbstractScene
from d123.geometry.bounding_box import BoundingBoxSE2
from d123.geometry.point import Point2D
from d123.geometry.transform.tranform_2d import translate_along_yaw
from d123.simulation.agents.abstract_agents import AbstractAgents


class ConstantVelocityAgents(AbstractAgents):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = False

    def __init__(self) -> None:
        """
        Initialize the constant velocity agents.
        """
        super().__init__()
        self._timestep_s: float = 0.1
        self._current_iteration: int = 0
        self._map_api: AbstractMap = None

        self._initial_target_agents: List[BoxDetection] = []

    @abstractmethod
    def reset(
        self,
        map_api: AbstractMap,
        target_agents: List[BoxDetection],
        non_target_agents: List[BoxDetection],
        scene: Optional[AbstractScene] = None,
    ) -> List[BoxDetection]:
        assert scene is None

        self._map_api = map_api
        self._current_iteration = 0
        self._initial_target_agents = [copy.deepcopy(agent) for agent in target_agents]
        return self._initial_target_agents

    def step(self, non_target_agents: List[BoxDetection]):
        self._current_iteration += 1

        time_delta_s = self._timestep_s * self._current_iteration
        current_target_agents = []
        for initial_agent in self._initial_target_agents:
            speed: float = float(initial_agent.velocity.vector_2d.magnitude)

            propagated_center = translate_along_yaw(initial_agent.center, Point2D(speed * time_delta_s, 0.0))
            propagated_bounding_box = BoundingBoxSE2(
                propagated_center,
                initial_agent.bounding_box_se3.length,
                initial_agent.bounding_box_se3.width,
            )
            propagated_agent: BoxDetectionSE2 = BoxDetectionSE2(
                metadata=initial_agent.metadata,
                bounding_box_se2=propagated_bounding_box,
                velocity=initial_agent.velocity,
            )
            current_target_agents.append(propagated_agent)

        return current_target_agents
