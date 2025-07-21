from asim.common.datatypes.vehicle_state.ego_state import DynamicStateSE2, EgoStateSE2
from asim.common.geometry.vector import Vector2D
from asim.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput


class ActionPlannerOutput(AbstractPlannerOutput):

    def __init__(self, acceleration: float, steering_rate: float, ego_state: EgoStateSE2):
        """
        Initializes the ActionTrajectory.
        :param acceleration: Longitudinal acceleration [m/s^2].
        :param steering_rate: Steering rate [rad/s].
        :param ego_state: Ego state at the start of the action.
        """

        self._acceleration = acceleration
        self._steering_rate = steering_rate
        self._ego_state = ego_state

    @property
    def dynamic_state_se2(self) -> DynamicStateSE2:

        return DynamicStateSE2(
            velocity=self._ego_state.dynamic_state_se2.velocity,
            acceleration=Vector2D(self._acceleration, 0.0),
            angular_velocity=self._ego_state.dynamic_state_se2.angular_velocity,
            tire_steering_rate=self._steering_rate,
            angular_acceleration=0.0,
        )
