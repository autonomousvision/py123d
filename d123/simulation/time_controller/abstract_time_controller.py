from __future__ import annotations

import abc
from typing import Tuple

from d123.conversion.scene.abstract_scene import AbstractScene
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


class AbstractTimeController(abc.ABC):
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
    def reset(self, scene: AbstractScene) -> SimulationIteration:
        """
        Reset the observation (all internal states should be reseted, if any).
        """

    @abc.abstractmethod
    def step(self) -> Tuple[SimulationIteration, bool]:
        """
        Advance to the next simulation iteration.
        :return: A tuple containing the next SimulationIteration and a boolean indicating if the simulation has reached its end.
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
