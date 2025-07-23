from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class AbstractTrajectoryBuilder(ABC):
    """Abstract class for building trajectories (nuPlan interface) in a Gym simulation environment."""

    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """
        Returns the action space of the gym environment.
        :return: gymnasium action space.
        """

    @abstractmethod
    def build_trajectory(
        self, action: npt.NDArray[np.float32], ego_state: EgoState, info: Dict[str, Any]
    ) -> AbstractTrajectory:
        """
        Builds a trajectory based on the action and the current ego state.
        :param action: Action taken by the agent, typically a numpy array.
        :param ego_state: Current state of the ego vehicle.
        :param info: Arbitrary information dictionary, for passing information between modules.
        :return: Trajectory object of nuPlan.
        """
