from dataclasses import dataclass

from d123.simulation.controller.abstract_controller import AbstractEgoController
from d123.simulation.observation.abstract_observation import AbstractObservation
from d123.simulation.time_controller.abstract_time_controller import AbstractTimeController


@dataclass
class Simulation2DSetup:
    """Setup class for constructing a 2D Simulation."""

    time_controller: AbstractTimeController
    observations: AbstractObservation
    ego_controller: AbstractEgoController

    def __post_init__(self) -> None:
        """Post-initialization sanity checks."""
        assert isinstance(
            self.time_controller, AbstractTimeController
        ), "Error: time_controller must inherit from AbstractTimeController!"
        assert isinstance(
            self.observations, AbstractObservation
        ), "Error: observations must inherit from AbstractObservation!"
        assert isinstance(
            self.ego_controller, AbstractEgoController
        ), "Error: ego_controller must inherit from AbstractEgoController!"
