from typing import Optional

from asim.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.controller.abstract_controller import AbstractEgoController
from asim.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel
from asim.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from asim.simulation.planning.planner_output.action_planner_output import ActionPlannerOutput
from asim.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class ActionController(AbstractEgoController):

    def __init__(self, scene: AbstractScene, motion_model: AbstractMotionModel):

        self._scene = scene
        self._motion_model = motion_model

        #  lazy loaded
        self._current_state: Optional[EgoStateSE2] = None

    def get_state(self) -> EgoStateSE2:
        """Inherited, see superclass."""
        if self._current_state is None:
            self._current_state = self._scene.get_ego_state_at_iteration(0).ego_state_se2
        return self._current_state

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._current_state = None

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoStateSE2,
        planner_output: AbstractPlannerOutput,
    ) -> None:
        """Inherited, see superclass."""

        assert isinstance(planner_output, ActionPlannerOutput)
        action: ActionPlannerOutput = planner_output

        sampling_time = next_iteration.time_point - current_iteration.time_point

        # Compute the dynamic state to propagate the model
        dynamic_state = action.dynamic_car_state

        # Propagate ego state using the motion model
        self._current_state = self._motion_model.propagate_state(
            state=ego_state,
            ideal_dynamic_state=dynamic_state,
            sampling_time=sampling_time,
        )
