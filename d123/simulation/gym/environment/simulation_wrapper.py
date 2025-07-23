# TODO: refactor and maybe move in environment wrapper
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import AbstractTrajectory
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.simulation import Simulation
from shapely.creation import linestrings
from shapely.geometry import LineString


class SimulationWrapper:
    """
    Helper object to wrap the nuPlan simulation and provide additional functionality.
    TODO:
        - Refactor this class.
        - Move route completion logic into the reward builder.
    """

    def __init__(self, simulation: Simulation):
        """
        Initializes the SimulationWrapper.
        :param simulation: Simulation object of nuPlan to wrap.I
        """

        self._simulation: Simulation = simulation
        self._route_completion: float = 0.0
        self._history_trajectories: List[AbstractTrajectory] = []

        # lazy loaded
        self._planner_initialization: Optional[PlannerInitialization]
        self._planner_input: Optional[PlannerInput]
        self._simulation_ego_states: List[EgoState] = []

    def initialize(self) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Initializes the simulation and returns the planner input and initialization.
        :return: Tuple containing the planner input and initialization according to the nuPlan interface.
        """
        self._planner_initialization = self._simulation.initialize()
        self._planner_input = self._simulation.get_planner_input()
        self._simulation_ego_states.append(self._planner_input.history.current_state[0])
        return self._planner_input, self._planner_initialization

    def step(self, trajectory: AbstractTrajectory) -> Tuple[PlannerInput, bool]:
        """
        Propagates the simulation and returns the new planner input.
        :return: Tuple containing the planner input and whether the simulation is running.
        """
        assert self._planner_initialization is not None, "SimulationManager: Call .initialize() first!"
        self._history_trajectories.append(trajectory)
        self._simulation.propagate(trajectory)
        self._planner_input = self._simulation.get_planner_input()
        self._simulation_ego_states.append(self._planner_input.history.current_state[0])
        output = self._planner_input, self.is_simulation_running()
        return output

    def is_simulation_running(self) -> bool:
        """
        Checks if the simulation is still running.
        :return: boolean.
        """
        iteration = self._simulation._time_controller.get_iteration().index
        num_iterations = self._simulation._time_controller.number_of_iterations()
        return iteration != num_iterations - 2

    @property
    def simulation(self) -> Simulation:
        """
        :return: Simulation used by the SimulationRunner
        """
        assert self._simulation is not None
        return self._simulation

    @property
    def history_trajectories(self) -> List[AbstractTrajectory]:
        """
        :return: Simulation used by the SimulationRunner
        """
        assert len(self._history_trajectories) > 0
        return self._history_trajectories

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario relative to the simulation.
        """
        assert self._simulation is not None
        return self._simulation.scenario

    @property
    def planner_initialization(self) -> PlannerInitialization:
        """
        :return: Get the current planner initialization object.
        """
        assert self._planner_initialization is not None
        return self._planner_initialization

    @property
    def simulation_ego_states(self) -> List[EgoState]:
        """
        :return: list of ego states from the simulation
        """
        return self._simulation_ego_states

    @property
    def current_planner_input(self) -> PlannerInput:
        """
        :return: Get the current planner initialization object.
        """
        assert self._planner_input is not None
        return self._planner_input

    @property
    def current_ego_state(self) -> EgoState:
        """
        :return: Current ego state from the simulation.
        """
        assert self._planner_input is not None
        ego_state, _ = self._planner_input.history.current_state
        return ego_state

    @property
    def initial_ego_state(self) -> EgoState:
        """
        :return: Initial ego state from the simulation.
        """
        return self.scenario.initial_ego_state

    @cached_property
    def ego_linestring(self) -> LineString:
        """
        Creates a linestring from the human ego states of the simulation.
        TODO: remove this function.
        :return: Shapely linestring of the human ego states.
        """
        ego_states = list(self.scenario.get_expert_ego_trajectory())
        ego_centers = np.array([ego_state.center.array for ego_state in ego_states])
        return linestrings(ego_centers)

    @property
    def route_completion(self):
        """
        TODO: remove this function. Move to reward builder.
        :return: The current route completion of the simulation [m].
        """
        return self._route_completion

    def update_route_completion(self, new_route_completion: float) -> None:
        """
        TODO: remove this function. Move to reward builder.
        """
        assert 0 <= new_route_completion <= 1
        self._route_completion = max(self._route_completion, new_route_completion)
