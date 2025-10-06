from typing import Optional, Tuple

from d123.datasets.scene.abstract_scene import AbstractScene
from d123.simulation.time_controller.abstract_time_controller import (
    AbstractTimeController,
)
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


class LogTimeController(AbstractTimeController):
    """
    Class handling simulation time and completion.
    """

    def __init__(self):
        """
        Initialize simulation control.
        """
        self._current_iteration_index: int = 0
        self._scene: Optional[AbstractScene] = None

    def reset(self, scene: AbstractScene) -> SimulationIteration:
        """Inherited, see superclass."""
        self._scene = scene
        self._current_iteration_index = 0
        return self.get_iteration()

    def get_iteration(self) -> SimulationIteration:
        """Inherited, see superclass."""
        scene_time = self._scene.get_timepoint_at_iteration(self._current_iteration_index)
        return SimulationIteration(time_point=scene_time, index=self._current_iteration_index)

    def step(self) -> Tuple[SimulationIteration, bool]:
        """Inherited, see superclass."""
        self._current_iteration_index += 1
        return self.get_iteration(), self.reached_end()

    def reached_end(self) -> bool:
        """Inherited, see superclass."""
        return self._current_iteration_index >= self.number_of_iterations() - 1

    def number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return self._scene.number_of_iterations
