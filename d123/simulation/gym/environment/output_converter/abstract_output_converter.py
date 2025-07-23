from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gymnasium import spaces


from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput


class AbstractOutputConverter(ABC):
    """Abstract class for building trajectories (nuPlan interface) in a Gym simulation environment."""

    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """
        Returns the action space of the gym environment.
        :return: gymnasium action space.
        """

    @abstractmethod
    def get_planner_output(
        self, action: npt.NDArray[np.float32], ego_state: EgoStateSE2, info: Dict[str, Any]
    ) -> AbstractPlannerOutput:
        """
        Builds a planner output based on the action and the current ego state.
        :param action: Action taken by the agent, typically a numpy array.
        :param ego_state: Current state of the ego vehicle.
        :param info: Arbitrary information dictionary, for passing information between modules.
        :return: Planner output object.
        """
