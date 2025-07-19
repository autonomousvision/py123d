from __future__ import annotations

import abc
from typing import Optional

from asim.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class AbstractSimulationTimeController(abc.ABC):
    """
    Generic simulation time manager.
    """

    @abc.abstractmethod
    def get_iteration(self) -> SimulationIteration:
        """
        Get the current simulation iteration.
        :return: Get the current simulation current_simulation_state and time point
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the observation (all internal states should be reseted, if any).
        """

    @abc.abstractmethod
    def next_iteration(self) -> Optional[SimulationIteration]:
        """
        Move to the next iteration and return its simulation iteration.
        Returns None if we have reached the end of the simulation.
        """

    @abc.abstractmethod
    def reached_end(self) -> bool:
        """
        Check if we have reached the end of the simulation.
        :return: Check whether simulation reached the end state.
        """

    @abc.abstractmethod
    def number_of_iterations(self) -> int:
        """
        The number of iterations the simulation should be running for
        :return: Number of iterations of simulation.
        """
