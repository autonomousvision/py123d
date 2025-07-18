from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
from carl_nuplan.planning.simulation.trajectory.action_trajectory import ActionTrajectory
from gymnasium import spaces
from nuplan.common.actor_state.ego_state import EgoState

from asim.simulation.gym.environment.trajectory_builder.abstract_trajectory_builder import (
    AbstractTrajectoryBuilder,
)
from asim.simulation.gym.policy.ppo.ppo_config import GlobalConfig


class ActionTrajectoryBuilder(AbstractTrajectoryBuilder):
    """
    Default action trajectory builder for training CaRL.
    TODO: Refactor this class
    NOTE @DanielDauner:
    We do an unclean hack here use an action (acceleration, steering) but package it into a Trajectory interface according to nuPlan.
    The nuPlan simulation strictly requires a trajectory. We use a OneStageController to skip the controller and directly propagate the bicycle model.
    You can create a new TrajectoryBuilder and SimulationBuilder to use a TwoStageController if you want to use a trajectory action.
    """

    def __init__(
        self,
        scale_max_acceleration: float = 2.4,
        scale_max_deceleration: float = 3.2,
        scale_max_steering_angle: float = 0.83775804096,
        clip_max_abs_steering_rate: Optional[float] = None,
        clip_max_abs_lon_jerk: Optional[float] = None,
        clip_max_abs_yaw_accel: Optional[float] = None,
        clip_angular_adjustment: bool = False,
        convert_low_pass_acceleration: bool = False,
        convert_low_pass_steering: bool = False,
        disable_reverse_driving: bool = True,
    ):
        """
        Initializes the ActionTrajectoryBuilder.
        :param scale_max_acceleration: max acceleration used for scaling the normed action [m/s^2], defaults to 2.4
        :param scale_max_deceleration: max deceleration (positive) used for scaling the normed action [m/s^2], defaults to 3.2
        :param scale_max_steering_angle: max absolute steering angle used for scaling the normed action [rad], defaults to 0.83775804096
        :param clip_max_abs_steering_rate: optional value to clip the steering rate [rad/s], defaults to None
        :param clip_max_abs_lon_jerk: optional value to clip the longitudinal jerk [m/s^3], defaults to None
        :param clip_max_abs_yaw_accel: optional value to clip the yaw acceleration [rad/s^2], defaults to None
        :param clip_angular_adjustment: Whether to adjust the longitudinal acceleration for lower rotation, defaults to False
        :param convert_low_pass_acceleration: Undo the low pass filtering of nuPlan's bicycle model, defaults to False
        :param convert_low_pass_steering: Undo the low pass filtering of nuPlan's bicycle model, defaults to False
        :param disable_reverse_driving: Whether to disable reverse driving with a controller, defaults to True
        """
        self._scale_max_acceleration = scale_max_acceleration  # [m/s^2]
        self._scale_max_deceleration = scale_max_deceleration  # [m/s^2]
        self._scale_max_steering_angle = scale_max_steering_angle  # [rad]

        self._clip_max_abs_steering_rate = clip_max_abs_steering_rate  # [rad/s]
        self._clip_max_abs_lon_jerk = clip_max_abs_lon_jerk  # [m/s^3]
        self._clip_max_abs_yaw_accel = clip_max_abs_yaw_accel  # [rad/s^2]
        self._clip_angular_adjustment = clip_angular_adjustment

        self._convert_low_pass_acceleration = convert_low_pass_acceleration
        self._convert_low_pass_steering = convert_low_pass_steering
        self._disable_reverse_driving = disable_reverse_driving

        self._config = GlobalConfig()
        self._dt_control = 0.1  # [s]
        self._accel_time_constant = 0.2  # [s]
        self._steering_angle_time_constant = 0.05  # [s]

    def get_action_space(self) -> spaces.Space:
        """Inherited, see superclass."""
        return spaces.Box(
            self._config.action_space_min,
            self._config.action_space_max,
            shape=(self._config.action_space_dim,),
            dtype=np.float32,
        )

    def build_trajectory(
        self, action: npt.NDArray[np.float32], ego_state: EgoState, info: Dict[str, Any]
    ) -> ActionTrajectory:
        """Inherited, see superclass."""
        assert len(action) == self._config.action_space_dim
        info["last_action"] = action

        action_acceleration_normed, action_steering_angle_normed = action

        target_steering_rate = self._scale_steering(
            action_steering_angle_normed,
            ego_state.tire_steering_angle,
        )
        target_acceleration = self._scale_acceleration(
            action_acceleration_normed,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
        )
        clipped_steering_rate = self._clip_steering(
            target_acceleration,
            target_steering_rate,
            ego_state,
        )
        clipped_acceleration = self._clip_acceleration(
            target_acceleration,
            clipped_steering_rate,
            ego_state,
        )

        store = info["store"] if "store" in info.keys() else None
        return ActionTrajectory(clipped_acceleration, clipped_steering_rate, ego_state, list(action), store)

    def _scale_steering(self, action_steering_angle_normed: float, current_steering_angle: float) -> float:
        """
        Scales the steering angle based on the action and current steering angle.
        :param action_steering_angle_normed: Normalized steering angle action, typically in [-1, 1].
        :param current_steering_angle: Current steering angle of the ego vehicle.
        :return: Scaled steering rate based on the action and current steering angle.
        """

        target_steering_angle = action_steering_angle_normed * self._scale_max_steering_angle
        if self._convert_low_pass_steering:
            factor = (self._dt_control + self._steering_angle_time_constant) / self._dt_control
            target_steering_angle = (target_steering_angle - current_steering_angle) * factor + current_steering_angle
        target_steering_rate = (target_steering_angle - current_steering_angle) / self._dt_control
        return target_steering_rate

    def _scale_acceleration(self, action_acceleration_normed: float, current_acceleration: float) -> float:
        """
        Scales the acceleration based on the action.
        :param action_acceleration_normed: Normalized acceleration action, typically in [-1, 1].
        :param current_acceleration: Current acceleration of the ego vehicle.
        :return: Scaled acceleration based on the action.
        """
        if action_acceleration_normed >= 0:
            target_acceleration = self._scale_max_acceleration * action_acceleration_normed
        else:
            target_acceleration = self._scale_max_deceleration * action_acceleration_normed
        if self._convert_low_pass_acceleration:
            factor = self._dt_control / (self._dt_control + self._accel_time_constant)
            target_acceleration = (target_acceleration - current_acceleration) / factor + current_acceleration
        return target_acceleration

    def _clip_acceleration(self, target_acceleration: float, target_steering_rate: float, ego_state: EgoState) -> float:
        """
        Clips the acceleration based on the target acceleration, steering rate, and current ego state.
        :param target_acceleration: Acceleration as targeted by the agent.
        :param target_steering_rate: Steering rate as targeted by the agent.
        :param ego_state: Current state of the ego vehicle.
        :return: Clipped acceleration based on the target acceleration and steering rate.
        """

        current_acceleration = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x

        if self._disable_reverse_driving:
            speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
            updated_speed = speed + target_acceleration * self._dt_control
            # * self._dt_control
            if updated_speed < 0:
                k_p, k_d = 1.0, 0.75
                error = -speed
                dt_error = -current_acceleration
                target_acceleration = k_p * error + k_d * dt_error

        if self._clip_max_abs_lon_jerk is not None:
            max_acceleration_change = self._clip_max_abs_lon_jerk * self._dt_control
            target_acceleration = np.clip(
                target_acceleration,
                current_acceleration - max_acceleration_change,
                current_acceleration + max_acceleration_change,
            )

        _max_acceleration = self._scale_max_acceleration
        if self._clip_angular_adjustment:
            rear_axle_to_center_dist = ego_state.car_footprint.rear_axle_to_center_dist

            next_point_velocity_x = (
                ego_state.dynamic_car_state.rear_axle_velocity_2d.x + target_acceleration * self._dt_control
            )
            next_point_tire_steering_angle = ego_state.tire_steering_angle + target_steering_rate * self._dt_control
            next_point_angular_velocity = (
                next_point_velocity_x
                * np.tan(next_point_tire_steering_angle)
                / ego_state.car_footprint.vehicle_parameters.wheel_base
            )
            next_point_angular_acceleration = (
                next_point_angular_velocity - ego_state.dynamic_car_state.angular_velocity
            ) / self._dt_control

            centripetal_acceleration_term = rear_axle_to_center_dist * (next_point_angular_velocity) ** 2
            angular_acceleration_term = rear_axle_to_center_dist * (next_point_angular_acceleration)
            _max_acceleration -= centripetal_acceleration_term + angular_acceleration_term

        target_acceleration = np.clip(target_acceleration, -self._scale_max_deceleration, _max_acceleration)
        return target_acceleration

    def _clip_steering(self, target_acceleration: float, target_steering_rate: float, ego_state: EgoState) -> float:
        """
        Clips the steering rate based on the target acceleration and current ego state.
        :param target_acceleration: Acceleration as targeted by the agent.
        :param target_steering_rate: Steering rate as targeted by the agent.
        :param ego_state: Current state of the ego vehicle.
        :return: Clipped steering rate based on the target acceleration and steering rate.
        """

        current_steering_angle = ego_state.tire_steering_angle
        target_steering_angle = current_steering_angle + target_steering_rate * self._dt_control

        if self._clip_max_abs_yaw_accel is not None:
            wheel_base = ego_state.car_footprint.vehicle_parameters.wheel_base
            target_velocity = (
                ego_state.dynamic_car_state.rear_axle_velocity_2d.x + target_acceleration * self._dt_control
            )

            current_angular_velocity = ego_state.dynamic_car_state.angular_velocity
            max_abs_yaw_velocity = self._clip_max_abs_yaw_accel * self._dt_control

            min_angular_velocity = current_angular_velocity - max_abs_yaw_velocity
            max_angular_velocity = current_angular_velocity + max_abs_yaw_velocity

            min_tire_steering_angle = np.arctan((min_angular_velocity * wheel_base) / target_velocity)
            max_tire_steering_angle = np.arctan((max_angular_velocity * wheel_base) / target_velocity)
            target_steering_angle = np.clip(target_steering_angle, min_tire_steering_angle, max_tire_steering_angle)
            target_steering_rate = (target_steering_angle - current_steering_angle) / self._dt_control

        if self._clip_max_abs_steering_rate is not None:
            target_steering_rate = np.clip(
                target_steering_rate,
                -self._clip_max_abs_steering_rate,
                self._clip_max_abs_steering_rate,
            )

        return target_steering_rate
