from typing import Optional, cast

from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from asim.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class StepSimulationTimeController(AbstractSimulationTimeController):
    """
    Class handling simulation time and completion.
    """

    def __init__(self, scene: AbstractScene):
        """
        Initialize simulation control.
        """
        self.current_iteration_index = 0
        self.scene = scene

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration_index = 0

    def get_iteration(self) -> SimulationIteration:
        """Inherited, see superclass."""
        scene_time = self.scene.get_time_point(self.current_iteration_index)
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
        return cast(int, self.scene.get_number_of_iterations())
