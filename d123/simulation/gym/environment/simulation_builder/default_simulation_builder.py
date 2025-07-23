import random

from carl_nuplan.planning.simulation.controller.one_stage_controller import OneStageController

# TODO: refactor for general motion models, observations, etc.
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)

from d123.simulation.gym.environment.simulation_builder.abstract_simulation_builder import (
    AbstractSimulationBuilder,
)


class DefaultSimulationBuilder(AbstractSimulationBuilder):
    """Default simulation builder for CaRL."""

    def __init__(self, agent_type: str = "tracks") -> None:
        """
        Initializes the DefaultSimulationBuilder.
        NOTE: Using "tracks" is by far the fastest option and recommended for testing and experimentation.
            The IDM implementation of nuPlan is very slow but could be improved if required.
        :param agent_type: whether to use tracks (log-replay), idm agents, a mixture, or no background, defaults to "tracks"
        """
        # TODO: use Literal typing.
        assert agent_type in ["tracks", "idm_agents", "mixed", "no_tracks"]

        self._agent_type = agent_type
        self._callback = None

        self._idm_agents_probability = 0.5
        self._history_buffer_duration = 1.0  # [s]

    def build_simulation(self, scenario: AbstractScenario) -> Simulation:
        """Inherited, see superclass."""
        simulation_setup = self._build_simulation_setup(scenario)
        return Simulation(
            simulation_setup=simulation_setup,
            callback=self._callback,
            simulation_history_buffer_duration=self._history_buffer_duration,
        )

    def _build_simulation_setup(self, scenario: AbstractScenario) -> SimulationSetup:
        """
        Helper function to build the simulation setup from a scenario.
        :param scenario: Scenario interface of nuPlan.
        :return: SimulationSetup object of nuPlan.
        """

        time_controller = StepSimulationTimeController(scenario)

        if self._agent_type == "mixed":
            use_idm_agents = random.random() < self._idm_agents_probability
            agent_type = "idm_agents" if use_idm_agents else "tracks"
        else:
            agent_type = self._agent_type

        if agent_type == "tracks":
            observations = TracksObservation(scenario)
        elif agent_type == "idm_agents":
            observations = IDMAgents(
                scenario=scenario,
                target_velocity=10,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.0,
                decel_max=2.0,
                open_loop_detections_types=[
                    "PEDESTRIAN",
                    "BARRIER",
                    "CZONE_SIGN",
                    "TRAFFIC_CONE",
                    "GENERIC_OBJECT",
                ],
                minimum_path_length=20,
                planned_trajectory_samples=None,
                planned_trajectory_sample_interval=None,
                radius=100,
            )

        motion_model = KinematicBicycleModel(get_pacifica_parameters())
        ego_controller = OneStageController(scenario, motion_model)

        return SimulationSetup(time_controller, observations, ego_controller, scenario)
