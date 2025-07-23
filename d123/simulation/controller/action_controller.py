from typing import Optional

from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.dataset.scene.abstract_scene import AbstractScene
from d123.simulation.controller.abstract_controller import AbstractEgoController
from d123.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel
from d123.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from d123.simulation.planning.planner_output.action_planner_output import ActionPlannerOutput
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


class ActionController(AbstractEgoController):

    def __init__(self, motion_model: AbstractMotionModel):

        self._motion_model = motion_model

        #  lazy loaded
        self._scene: Optional[AbstractScene] = None
        self._current_state: Optional[EgoStateSE2] = None

    def get_state(self) -> EgoStateSE2:
        """Inherited, see superclass."""
        if self._current_state is None:
            self._current_state = self._scene.get_ego_state_at_iteration(0).ego_state_se2
        return self._current_state

    def reset(self, scene: AbstractScene) -> EgoStateSE2:
        """Inherited, see superclass."""
        self._current_state = None
        self._scene = scene
        return self.get_state()

    def step(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoStateSE2,
        planner_output: AbstractPlannerOutput,
    ) -> EgoStateSE2:
        """Inherited, see superclass."""

        assert isinstance(planner_output, ActionPlannerOutput)
        action: ActionPlannerOutput = planner_output

        # Compute the dynamic state to propagate the model
        dynamic_state = action.dynamic_state_se2

        # Propagate ego state using the motion model
        self._current_state = self._motion_model.step(
            ego_state=ego_state,
            ideal_dynamic_state=dynamic_state,
            next_timepoint=next_iteration.time_point,
        )
        return self._current_state
