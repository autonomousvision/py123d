from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from d123.simulation.gym.environment.simulation_wrapper import SimulationWrapper


class AbstractRewardBuilder(ABC):
    """Abstract class for building rewards in a Gym simulation environment."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the reward builder to its initial state."""

    @abstractmethod
    def build_reward(self, simulation_wrapper: SimulationWrapper, info: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """
        Build the reward based on the current simulation state and additional information.
        :param simulation_wrapper: Wrapper object containing complete nuPlan simulation.
        :param info: Arbitrary information dictionary, for passing information between modules.
        :return: A tuple containing:
            - reward: The calculated reward value.
            - termination: Whether the simulation terminates in the current step.
            - truncation: Whether the simulation is truncated in the current step.
        """
