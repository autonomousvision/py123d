"""
NOTE @DanielDauner:

Learning comfortable driving behavior—according to nuPlan's comfort metrics proved to be challenging for a PPO policy operating directly in action space.
In contrast, the default nuPlan controllers (e.g., LQR, iLQR) explicitly optimize for comfort, so trajectory-based policies tend to achieve high comfort scores.
For example, Gigaflow generates full agent rollouts as trajectories and then uses the nuPlan controller to execute them, which likely mitigates comfort issues.

Designing effective comfort reward terms (see below) was one of the key challenges in adapting CaRL to nuPlan. A few more comments:
- nuPlan's comfort metrics apply Savitzky-Golay filtering, which can be noisy with short history windows. This also requires a min history length of 4.
- The metrics compute accelerations and velocities using the agent's center, not the rear axle. This mismatch previously introduced bugs in our PDM planners.
- The PPO policy requires a very large number of environment steps to appropriately balance the comfort term against others (e.g., collision avoidance).
- We spent a lot of time trying to make the comfort term work and expected bugs in the reward. In hindsight, we did not train for enough steps required for the comfort terms.

All comfort-related reward terms remain in the codebase but may be refactored in the future. We used `calculate_kinematics_comfort`.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from carl_nuplan.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
from carl_nuplan.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics_debug import (
    ego_is_comfortable_debug,
)
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)

from asim.simulation.gym.environment.simulation_wrapper import SimulationWrapper


def calculate_action_delta_comfort(simulation_wrapper: SimulationWrapper, max_change: float = 0.25) -> float:
    """
    Calculate the comfort score based on the change in actions between the last two time steps.
    :param simulation_wrapper: Wrapper object containing complete nuPlan simulation.
    :param max_change: max change considered comfortable in action space, defaults to 0.25
    :return: float values between 0.0 and 1.0, where 1.0 is most comfortable and 0.0 is least comfortable.
    """
    history_trajectories = simulation_wrapper.history_trajectories
    if len(history_trajectories) >= 2:

        current_action = simulation_wrapper.history_trajectories[-1]._raw_action
        previous_action = simulation_wrapper.history_trajectories[-2]._raw_action
        comfort = np.abs(np.array(current_action) - np.array(previous_action)) > max_change

        if np.any(comfort):
            return 0.5

    return 1.0


def calculate_kinematics_comfort(simulation_wrapper: SimulationWrapper) -> Tuple[float, npt.NDArray[np.bool_]]:
    """
    Calculate the comfort score based on the six kinematic metrics of nuPlan.
    NOTE: uses the debugged version of the comfort metrics using the center coordinate of the ego vehicle.
    :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
    :return: Whether the ego vehicle is comfortable and the comfort scores.
    """
    history_ego_states = simulation_wrapper.simulation_ego_states
    is_comfortable: npt.NDArray[np.bool_] = np.zeros((6), dtype=np.bool_)
    comfort_score: float = 1.0

    if len(history_ego_states) >= 4:
        history_states_array = ego_states_to_state_array(history_ego_states)[None, ...]
        time_points = np.array(
            [ego_state.time_point.time_s for ego_state in history_ego_states],
            dtype=np.float64,
        )
        is_comfortable = ego_is_comfortable_debug(history_states_array, time_points)[0]
        comfort_score = is_comfortable.sum() / len(is_comfortable)

    return comfort_score, is_comfortable


def calculate_kinematics_history_comfort(simulation_wrapper: SimulationWrapper) -> Tuple[float, npt.NDArray[np.bool_]]:
    """
    Calculate the comfort score based on the six kinematic metrics of nuPlan. Includes the ego history for calculation.
    NOTE: Adds the ego history of the logs to the comfort metrics. Was not relevant.
    :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
    :return: Whether the ego vehicle is comfortable and the comfort scores.
    """
    history_ego_states = simulation_wrapper.current_planner_input.history.ego_states
    assert len(history_ego_states) >= 4
    is_comfortable: npt.NDArray[np.bool_] = np.zeros((6), dtype=np.bool_)

    history_states_array = ego_states_to_state_array(history_ego_states)[None, ...]
    time_points = np.array(
        [ego_state.time_point.time_s for ego_state in history_ego_states],
        dtype=np.float64,
    )
    is_comfortable = ego_is_comfortable_debug(history_states_array, time_points)[0]
    comfort_score = is_comfortable.sum() / len(is_comfortable)

    return comfort_score, is_comfortable


def calculate_kinematics_comfort_legacy(simulation_wrapper: SimulationWrapper) -> Tuple[float, npt.NDArray[np.bool_]]:
    """
    Calculate the comfort score based on the six kinematic metrics of nuPlan.
    NOTE: Uses the rear-axle instead of center coordinate of the ego vehicle. Leads to slight mismatch to nuPlan metric.
    :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
    :return: Whether the ego vehicle is comfortable and the comfort scores.
    """
    history_ego_states = simulation_wrapper.simulation_ego_states
    is_comfortable: npt.NDArray[np.bool_] = np.zeros((6), dtype=np.bool_)

    if len(history_ego_states) >= 4:
        history_states_array = ego_states_to_state_array(history_ego_states)[None, ...]
        time_points = np.array(
            [ego_state.time_point.time_s for ego_state in history_ego_states],
            dtype=np.float64,
        )
        is_comfortable = ego_is_comfortable(history_states_array, time_points)[0]
        if not is_comfortable.all():
            return 0.5, is_comfortable

    return 1.0, is_comfortable


def calculate_kinematics_comfort_fixed(simulation_wrapper: SimulationWrapper) -> Tuple[float, npt.NDArray[np.bool_]]:
    """
    Calculate the comfort score based on the six kinematic metrics of nuPlan.
    NOTE: Ignores certain jerk metrics that are noisy initially.
    :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
    :return: Whether the ego vehicle is comfortable and the comfort scores.
    """
    history_ego_states = simulation_wrapper.simulation_ego_states
    is_comfortable: npt.NDArray[np.bool_] = np.zeros((6), dtype=np.bool_)

    if len(history_ego_states) >= 4:
        history_states_array = ego_states_to_state_array(history_ego_states)[None, ...]
        time_points = np.array(
            [ego_state.time_point.time_s for ego_state in history_ego_states],
            dtype=np.float64,
        )
        is_comfortable = ego_is_comfortable_debug(history_states_array, time_points)[0]

        if len(history_ego_states) < 15:
            is_comfortable[2] = True  # NOTE: jerk metric is trash in first few frames
            is_comfortable[3] = True  # NOTE: lon jerk metric is trash in first few frames

        if not is_comfortable.all():
            return 0.5, is_comfortable

    return 1.0, is_comfortable
