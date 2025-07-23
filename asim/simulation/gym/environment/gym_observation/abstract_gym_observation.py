from abc import ABC, abstractmethod
from typing import Any, Dict

from gymnasium import spaces

from asim.simulation.planning.abstract_planner import PlannerInitialization, PlannerInput


class AbstractGymObservation(ABC):
    """Abstract class for building observations in a gym environment."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the observation builder."""

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Get the observation space of the environment.
        :return: Observation space as a gymnasium Space.
        """

    @abstractmethod
    def get_gym_observation(
        self,
        planner_input: PlannerInput,
        planner_initialization: PlannerInitialization,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build an observation from the planner input and initialization.
        :param planner_input: Planner input as defined in the asim interface.
        :param planner_initialization: Planner initialization as defined in the asim interface.
        :param info: Arbitrary information dictionary, for passing information between modules.
        :return: Observation as a named dictionary.
        """
