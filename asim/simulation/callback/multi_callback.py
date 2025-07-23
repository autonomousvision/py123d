from typing import List

from asim.simulation.callback.abstract_callback import AbstractCallback
from asim.simulation.history.simulation_history import Simulation2DHistory, Simulation2DHistorySample
from asim.simulation.planning.abstract_planner import AbstractPlanner
from asim.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from asim.simulation.simulation_2d_setup import Simulation2DSetup


class MultiCallback(AbstractCallback):
    """
    This class simply calls many callbacks for simplified code.
    """

    def __init__(self, callbacks: List[AbstractCallback]):
        """
        Initialize with multiple callbacks.
        :param callbacks: all callbacks that will be called sequentially.
        """
        self._callbacks = callbacks

    @property
    def callbacks(self) -> List[AbstractCallback]:
        """
        Property to access callbacks.
        :return: list of callbacks this MultiCallback runs.
        """
        return self._callbacks

    def on_initialization_start(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_initialization_start(setup, planner)

    def on_initialization_end(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_initialization_end(setup, planner)

    def on_step_start(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_step_start(setup, planner)

    def on_step_end(
        self, setup: Simulation2DSetup, planner: AbstractPlanner, sample: Simulation2DHistorySample
    ) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_step_end(setup, planner, sample)

    def on_planner_start(self, setup: Simulation2DSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_planner_start(setup, planner)

    def on_planner_end(
        self, setup: Simulation2DSetup, planner: AbstractPlanner, planner_output: AbstractPlannerOutput
    ) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_planner_end(setup, planner, planner_output)

    def on_simulation_start(self, setup: Simulation2DSetup) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_simulation_start(setup)

    def on_simulation_end(
        self, setup: Simulation2DSetup, planner: AbstractPlanner, history: Simulation2DHistory
    ) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_simulation_end(setup, planner, history)
