from abc import ABC, abstractmethod

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation import Simulation


class AbstractSimulationBuilder(ABC):
    """Abstract class for building a nuPlan simulation object, which includes background traffic, etc."""

    @abstractmethod
    def build_simulation(self, scenario: AbstractScenario) -> Simulation:
        """
        Builds a nuPlan simulation object.
        :param scenario: Scenario interface of nuPlan.
        :return: Simulation object of nuPlan.
        """
