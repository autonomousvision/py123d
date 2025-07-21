from typing import Optional

from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.time_controller.abstract_time_controller import (
    AbstractTimeController,
)
from asim.simulation.time_controller.simulation_iteration import SimulationIteration


class LogTimeController(AbstractTimeController):
    """
    Class handling simulation time and completion.
    """

    def __init__(self):
        """
        Initialize simulation control.
        """
        self.current_iteration_index: int = 0
        self._scene: Optional[AbstractScene] = None

    def reset(self, scene: AbstractScene) -> SimulationIteration:
        """Inherited, see superclass."""
        self.current_iteration_index = 0
        self._scene = scene
        return self.get_iteration()

    def get_iteration(self) -> SimulationIteration:
        """Inherited, see superclass."""
        scene_time = self._scene.get_timepoint_at_iteration(self.current_iteration_index)
        return SimulationIteration(time_point=scene_time, index=self.current_iteration_index)

    def next_iteration(self) -> Optional[SimulationIteration]:
        """Inherited, see superclass."""
        self.current_iteration_index += 1
        return None if self.reached_end() else self.get_iteration()

    def reached_end(self) -> bool:
        """Inherited, see superclass."""
        return self.current_iteration_index >= self.number_of_iterations() - 1

    def number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return self._scene.get_number_of_iterations()
