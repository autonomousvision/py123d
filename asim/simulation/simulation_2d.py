from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Type

from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.callback.abstract_callback import AbstractCallback
from asim.simulation.callback.multi_callback import MultiCallback
from asim.simulation.history.simulation_history import Simulation2DHistory, Simulation2DHistorySample
from asim.simulation.history.simulation_history_buffer import Simulation2DHistoryBuffer
from asim.simulation.planning.abstract_planner import PlannerInitialization, PlannerInput
from asim.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from asim.simulation.simulation_2d_setup import Simulation2DSetup

logger = logging.getLogger(__name__)


class Simulation2D:

    def __init__(
        self,
        simulation_2d_setup: Simulation2DSetup,
        callback: Optional[AbstractCallback] = None,
    ):

        # Store all engines
        self._setup = simulation_2d_setup

        # Proxy
        self._time_controller = simulation_2d_setup.time_controller
        self._ego_controller = simulation_2d_setup.ego_controller
        self._observations = simulation_2d_setup.observations
        self._scene = simulation_2d_setup.scene
        self._callback = MultiCallback([]) if callback is None else callback

        # History where the steps of a simulation are stored
        self._history = Simulation2DHistory(self._scene.map_api)

        # The + 1 here is to account for duration. For example, 20 steps at 0.1s starting at 0s will have a duration
        # of 1.9s. At 21 steps the duration will achieve the target 2s duration.
        self._history_buffer: Optional[Simulation2DHistoryBuffer] = None

        # Flag that keeps track whether simulation is still running
        self._is_simulation_running = True

    def __reduce__(self) -> Tuple[Type[Simulation2D], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._setup, self._callback)

    def is_simulation_running(self) -> bool:
        """
        Check whether a simulation reached the end
        :return True if simulation hasn't reached the end, otherwise false.
        """
        return not self._time_controller.reached_end() and self._is_simulation_running

    def reset(self) -> None:
        """
        Reset all internal states of simulation.
        """
        # Clear created log
        self._history.reset()

        # Reset all simulation internal members
        self._setup.reset()

        # Clear history buffer
        self._history_buffer = None

        # Restart simulation
        self._is_simulation_running = True

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scene
        self._history_buffer = Simulation2DHistoryBuffer.initialize_from_scene(
            self._history_buffer_size, self._scene, self._observations.observation_type()
        )

        # Initialize observations
        self._observations.initialize()

        # Add the current state into the history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

        # Return the planner initialization structure for this simulation
        return PlannerInitialization(
            route_lane_group_ids=self._scene.get_route_lane_group_ids(0),
            map_api=self._scene.map_api,
        )

    def get_planner_input(self) -> PlannerInput:
        """
        Construct inputs to the planner for the current iteration step
        :return Inputs to the planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, stepping can not be performed!")

        # Extract current state
        iteration = self._time_controller.get_iteration()

        # Extract traffic light status data
        traffic_light_data = list(self._scene.get_traffic_light_status_at_iteration(iteration.index))
        logger.debug(f"Executing {iteration.index}!")
        return PlannerInput(iteration=iteration, history=self._history_buffer, traffic_light_data=traffic_light_data)

    def propagate(self, planner_output: AbstractPlannerOutput) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, simulation can not be propagated!")

        # Measurements
        iteration = self._time_controller.get_iteration()
        ego_state, observation = self._history_buffer.current_state

        # Add new sample to history
        logger.debug(f"Adding to history: {iteration.index}")
        self._history.add_sample(Simulation2DHistorySample(iteration, ego_state, planner_output, observation))

        # Propagate state to next iteration
        next_iteration = self._time_controller.next_iteration()

        # Propagate state
        if next_iteration:
            self._ego_controller.update_state(iteration, next_iteration, ego_state, planner_output)
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)
        else:
            self._is_simulation_running = False

        # Append new state into history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

    @property
    def scene(self) -> AbstractScene:
        """
        :return: used scene in this simulation.
        """
        return self._scene

    @property
    def setup(self) -> Simulation2DSetup:
        """
        :return: Setup for this simulation.
        """
        return self._setup

    @property
    def callback(self) -> AbstractCallback:
        """
        :return: Callback for this simulation.
        """
        return self._callback

    @property
    def history(self) -> Simulation2DHistory:
        """
        :return History from the simulation.
        """
        return self._history

    @property
    def history_buffer(self) -> Simulation2DHistoryBuffer:
        """
        :return SimulationHistoryBuffer from the simulation.
        """
        if self._history_buffer is None:
            raise RuntimeError(
                "_history_buffer is None. Please initialize the buffer by calling Simulation.initialize()"
            )
        return self._history_buffer
