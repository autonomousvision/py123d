from abc import ABC, abstractmethod

from asim.simulation.history.simulation_history import Simulation2DHistory, Simulation2DHistorySample
from asim.simulation.planning.abstract_planner import AbstractPlanner
from asim.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from asim.simulation.simulation_2d_setup import Simulation2DSetup


class AbstractCallback(ABC):
    """
    Base class for simulation callbacks.
    """

    @abstractmethod
    def on_initialization_start(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """
        Called when initialization of simulation starts.
        :param setup: simulation setup
        :param planner: planner before initialization
        """

    @abstractmethod
    def on_initialization_end(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """
        Called when initialization of simulation ends.
        :param setup: simulation setup
        :param planner: planner after initialization
        """

    @abstractmethod
    def on_step_start(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """
        Called when simulation step starts.
        :param setup: simulation setup
        :param planner: planner at start of a step
        """

    @abstractmethod
    def on_step_end(
        self, setup: Simulation2DSetup, planner: AbstractPlanner, sample: Simulation2DHistorySample
    ) -> None:
        """
        Called when simulation step ends.
        :param setup: simulation setup
        :param planner: planner at end of a step
        :param sample: result of a step
        """

    @abstractmethod
    def on_planner_start(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """
        Called when planner starts to compute trajectory.
        :param setup: simulation setup
        :param planner: planner before planner.compute_trajectory() is called
        """

    @abstractmethod
    def on_planner_end(
        self, setup: Simulation2DSetup, planner: AbstractPlanner, planner_output: AbstractPlannerOutput
    ) -> None:
        pass

    @abstractmethod
    def on_simulation_start(self, setup: Simulation2DSetup) -> None:
        """
        Called when simulation starts.
        :param setup: simulation setup
        """

    @abstractmethod
    def on_simulation_end(
        self, setup: Simulation2DSetup, planner: AbstractPlanner, history: Simulation2DHistory
    ) -> None:
        """
        Called when simulation ends.
        :param setup: simulation setup
        :param planner: planner when simulation ends
        :param history: resulting from simulation
        """
