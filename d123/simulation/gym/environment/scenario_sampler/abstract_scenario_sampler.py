from abc import ABC, abstractmethod
from typing import List, Optional

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario


class AbstractScenarioSampler(ABC):
    """Abstract class for sampling scenarios in a Gym simulation environment."""

    @abstractmethod
    def sample(self, seed: Optional[int] = None) -> AbstractScenario:
        """
        Samples a single scenario.
        :param seed: Optional seed used for sampling, defaults to None
        :return: Scenario interface of nuPlan.
        """

    @abstractmethod
    def sample_batch(self, batch_size: int, seed: Optional[int] = None) -> List[AbstractScenario]:
        """
        Samples a batch of scenarios.
        :param batch_size: number of scenarios to sample.
        :param seed: Optional seed used for sampling, defaults to None
        :return: List of scenario interfaces of nuPlan.
        """
