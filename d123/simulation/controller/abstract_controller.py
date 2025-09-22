import abc

from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


class AbstractEgoController(abc.ABC):
    """
    Interface for generic ego controllers.
    """

    @abc.abstractmethod
    def get_state(self) -> EgoStateSE2:
        """
        Returns the current ego state.
        :return: The current ego state.
        """

    @abc.abstractmethod
    def reset(self, scene: AbstractScene) -> EgoStateSE2:
        """
        Reset the observation (all internal states should be reseted, if any).
        """

    @abc.abstractmethod
    def step(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoStateSE2,
        planner_output: AbstractPlannerOutput,
    ) -> EgoStateSE2:
        """
        Update the ego state based on the planner output and the current state.
        :param current_iteration: The current simulation iteration.
        :param next_iteration: The next simulation iteration after propagation.
        :param ego_state: The current ego state.
        :param planner_output: The output of a planner, e.g. action or trajectory.
        :return: The updated ego state.
        """
