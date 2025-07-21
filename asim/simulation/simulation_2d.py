from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Type

from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.callback.abstract_callback import AbstractCallback
from asim.simulation.callback.multi_callback import MultiCallback
from asim.simulation.controller.abstract_controller import AbstractEgoController
from asim.simulation.history.simulation_history import Simulation2DHistory, Simulation2DHistorySample
from asim.simulation.history.simulation_history_buffer import Simulation2DHistoryBuffer
from asim.simulation.observation.abstract_observation import AbstractObservation
from asim.simulation.planning.abstract_planner import PlannerInitialization, PlannerInput
from asim.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from asim.simulation.time_controller.abstract_time_controller import AbstractTimeController

logger = logging.getLogger(__name__)


class Simulation2D:

    def __init__(
        self,
        time_controller: AbstractTimeController,
        observations: AbstractObservation,
        ego_controller: AbstractEgoController,
        callback: Optional[AbstractCallback] = None,
    ):

        self._time_controller = time_controller
        self._observations = observations
        self._ego_controller = ego_controller
        self._callback = MultiCallback([]) if callback is None else callback

        # History where the steps of a simulation are stored
        self._history = Simulation2DHistory()

        # The + 1 here is to account for duration. For example, 20 steps at 0.1s starting at 0s will have a duration
        # of 1.9s. At 21 steps the duration will achieve the target 2s duration.
        self._history_buffer: Optional[Simulation2DHistoryBuffer] = None

        # Flag that keeps track whether simulation is still running
        self._is_simulation_running = True

        # Lazy loaded in `.reset()` method
        self._scene: Optional[AbstractScene] = None

    def __reduce__(self) -> Tuple[Type[Simulation2D], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (
            self._time_controller,
            self._observations,
            self._ego_controller,
            self._callback,
        )

    def is_simulation_running(self) -> bool:
        """
        Check whether a simulation reached the end
        :return True if simulation hasn't reached the end, otherwise false.
        """
        return not self._time_controller.reached_end() and self._is_simulation_running

    def reset(self, scene: AbstractScene) -> Tuple[PlannerInitialization, PlannerInput]:
        """
        Reset all internal states of simulation.
        """

        # 1. Reset the scene object
        self._scene = scene

        # 2. Reset history and setup
        self._history.reset()
        simulation_iteration = self._time_controller.reset(self._scene)
        observation = self._observations.reset(self._scene)
        ego_state = self._ego_controller.reset(self._scene)

        # 3. Reinitialize history buffer
        self._history_buffer = Simulation2DHistoryBuffer.initialize_from_scene(
            self._scene.get_number_of_history_iterations(), self._scene, self._observations.recording_type()
        )
        self._history_buffer.append(ego_state, observation)

        # Restart simulation
        self._is_simulation_running = True
        planner_initialization = PlannerInitialization(
            route_lane_group_ids=self._scene.get_route_lane_group_ids(0),
            map_api=self._scene.map_api,
        )
        planner_input = PlannerInput(iteration=simulation_iteration, history=self._history_buffer)

        return planner_initialization, planner_input

    def step(self, planner_output: AbstractPlannerOutput) -> PlannerInput:

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
    def scene(self) -> Optional[AbstractScene]:
        """
        :return: used scene in this simulation.
        """
        return self._scene

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
