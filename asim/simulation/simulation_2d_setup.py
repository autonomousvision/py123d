from dataclasses import dataclass

from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.controller.abstract_controller import AbstractEgoController
from asim.simulation.observation.abstract_observation import AbstractObservation
from asim.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)

# from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
# from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
# from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
# from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
# from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
#     AbstractSimulationTimeController,
# )


@dataclass
class Simulation2DSetup:
    """Setup class for constructing a 2D Simulation."""

    time_controller: AbstractSimulationTimeController
    observations: AbstractObservation
    ego_controller: AbstractEgoController
    scene: AbstractScene

    def __post_init__(self) -> None:
        """Post-initialization sanity checks."""
        # Other checks
        assert isinstance(
            self.time_controller, AbstractSimulationTimeController
        ), "Error: simulation_time_controller must inherit from AbstractSimulationTimeController!"
        assert isinstance(
            self.observations, AbstractObservation
        ), "Error: observations must inherit from AbstractObservation!"
        assert isinstance(
            self.ego_controller, AbstractEgoController
        ), "Error: ego_controller must inherit from AbstractEgoController!"

    def reset(self) -> None:
        """
        Reset all simulation controllers
        """
        self.observations.reset()
        self.ego_controller.reset()
        self.time_controller.reset()
