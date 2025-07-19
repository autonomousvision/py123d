import abc

from asim.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from asim.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from asim.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


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
    def reset(self) -> None:
        """
        Reset the observation (all internal states should be reseted, if any).
        """

    @abc.abstractmethod
    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoStateSE2,
        planner_output: AbstractPlannerOutput,
    ) -> None:
        """
        Update ego's state from current iteration to next iteration.

        :param current_iteration: The current simulation iteration.
        :param next_iteration: The desired next simulation iteration.
        :param ego_state: The current ego state.
        :param planner_output: The output of a planner, e.g. action or trajectory.
        """
