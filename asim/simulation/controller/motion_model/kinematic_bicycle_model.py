import numpy as np

# from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
# from nuplan.common.actor_state.ego_state import EgoState, EgoStateDot
# from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
# from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.compute import principal_value

from asim.common.datatypes.time.time_point import TimeDuration, TimePoint
from asim.common.datatypes.vehicle_state.ego_state import DynamicStateSE2, EgoStateSE2
from asim.common.geometry.base import StateSE2
from asim.common.geometry.vector import Vector2D
from asim.simulation.motion_model.abstract_motion_model import AbstractMotionModel


def forward_integrate(init: float, delta: float, sampling_duration: TimeDuration) -> float:
    return float(init + delta * sampling_duration.time_s)


class KinematicBicycleModel(AbstractMotionModel):

    def __init__(
        self,
        max_steering_angle: float = np.pi / 3,
        acceleration_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
        acceleration_low_pass_filter: bool = True,
        steering_angle_low_pass_filter: bool = True,
    ):
        self._max_steering_angle = max_steering_angle
        self._acceleration_time_constant = acceleration_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant
        self._acceleration_low_pass_filter = acceleration_low_pass_filter
        self._steering_angle_low_pass_filter = steering_angle_low_pass_filter

    def get_state_dot(self, state: EgoStateSE2) -> EgoStateSE2:

        long_speed = state.dynamic_state_se2.velocity.x
        wheel_base = state.vehicle_parameters.wheel_base
        x_dot = long_speed * np.cos(state.rear_axle.yaw)
        y_dot = long_speed * np.sin(state.rear_axle.yaw)
        yaw_dot = long_speed * np.tan(state.tire_steering_angle) / wheel_base

        return EgoStateSE2.from_rear_axle(
            rear_axle_se2=StateSE2(x=x_dot, y=y_dot, yaw=yaw_dot),
            dynamic_state_se2=DynamicStateSE2(
                velocity=state.dynamic_state_se2.acceleration,
                acceleration=Vector2D(0.0, 0.0),
                angular_velocity=0.0,
                tire_steering_rate=0.0,
            ),
            vehicle_parameters=state.vehicle_parameters,
            time_point=state.timepoint,
            tire_steering_angle=state.dynamic_state_se2.tire_steering_rate,
        )

    def _update_commands(
        self,
        ego_state: EgoStateSE2,
        ideal_dynamic_state: DynamicStateSE2,
        step_duration: TimeDuration,
    ) -> EgoStateSE2:

        dt_control = step_duration.time_s
        long_acceleration = ego_state.dynamic_state_se2.acceleration.x
        tire_steering_angle = ego_state.tire_steering_angle

        ideal_long_acceleration = ideal_dynamic_state.acceleration.x
        ideal_steering_angle = dt_control * ideal_dynamic_state.tire_steering_rate + tire_steering_angle

        if self._acceleration_low_pass_filter:
            updated_long_acceleration = (
                dt_control
                / (dt_control + self._acceleration_time_constant)
                * (ideal_long_acceleration - long_acceleration)
                + long_acceleration
            )
        else:
            updated_long_acceleration = ideal_long_acceleration

        if self._steering_angle_low_pass_filter:
            updated_steering_angle = (
                dt_control
                / (dt_control + self._steering_angle_time_constant)
                * (ideal_steering_angle - tire_steering_angle)
                + tire_steering_angle
            )
        else:
            updated_steering_angle = ideal_steering_angle

        updated_steering_rate = (updated_steering_angle - tire_steering_angle) / dt_control
        dynamic_state = DynamicStateSE2(
            velocity=ego_state.dynamic_state_se2.velocity,
            acceleration=Vector2D(updated_long_acceleration, 0.0),
            angular_velocity=0.0,
            tire_steering_rate=updated_steering_rate,
            angular_acceleration=0.0,
        )
        propagating_state = EgoStateSE2(
            center_se2=ego_state.center_se2,
            dynamic_state_se2=dynamic_state,
            vehicle_parameters=ego_state.vehicle_parameters,
            timepoint=ego_state.timepoint,
            tire_steering_angle=ego_state.tire_steering_angle,
        )
        return propagating_state

    def step(
        self,
        ego_state: EgoStateSE2,
        ideal_dynamic_state: DynamicStateSE2,
        sampling_time: TimePoint,
    ) -> EgoStateSE2:

        vehicle_parameters = ego_state.vehicle_parameters

        # step_duration = ego_state.timepoint.diff(sampling_time)
        step_duration = sampling_time.diff(ego_state.timepoint)
        propagating_state = self._update_commands(ego_state, ideal_dynamic_state, step_duration)

        # Compute state derivatives
        state_dot = self.get_state_dot(propagating_state)

        # Integrate position and heading
        next_x = forward_integrate(propagating_state.rear_axle.x, state_dot.rear_axle.x, step_duration)
        next_y = forward_integrate(propagating_state.rear_axle.y, state_dot.rear_axle.y, step_duration)
        next_yaw = forward_integrate(propagating_state.rear_axle.yaw, state_dot.rear_axle.yaw, step_duration)
        # Wrap angle between [-pi, pi]
        next_yaw = principal_value(next_yaw)

        # Compute rear axle velocity in car frame
        next_point_velocity_x = forward_integrate(
            propagating_state.dynamic_state_se2.velocity.x,
            state_dot.dynamic_state_se2.velocity.x,
            step_duration,
        )
        next_point_velocity_y = 0.0  # Lateral velocity is always zero in kinematic bicycle model

        # Integrate steering angle and clip to bounds
        next_point_tire_steering_angle = np.clip(
            forward_integrate(propagating_state.tire_steering_angle, state_dot.tire_steering_angle, step_duration),
            -self._max_steering_angle,
            self._max_steering_angle,
        )

        # Compute angular velocity
        next_point_angular_velocity = (
            next_point_velocity_x * np.tan(next_point_tire_steering_angle) / vehicle_parameters.wheel_base
        )

        rear_axle_accel = [
            state_dot.dynamic_state_se2.velocity.x,
            state_dot.dynamic_state_se2.velocity.y,
        ]
        angular_accel = (
            next_point_angular_velocity - ego_state.dynamic_state_se2.angular_velocity
        ) / step_duration.time_s

        return EgoStateSE2.from_rear_axle(
            rear_axle_se2=StateSE2(next_x, next_y, next_yaw),
            dynamic_state_se2=DynamicStateSE2(
                velocity=Vector2D(next_point_velocity_x, next_point_velocity_y),
                acceleration=Vector2D(rear_axle_accel[0], rear_axle_accel[1]),
                tire_steering_rate=state_dot.tire_steering_angle,
                angular_velocity=next_point_angular_velocity,
                angular_acceleration=angular_accel,
            ),
            vehicle_parameters=vehicle_parameters,
            time_point=sampling_time,
            tire_steering_angle=float(next_point_tire_steering_angle),
        )
