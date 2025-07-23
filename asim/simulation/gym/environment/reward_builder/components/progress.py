from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    PerFrameProgressAlongRouteComputer,
)
from nuplan.planning.metrics.utils.route_extractor import (
    RouteRoadBlockLinkedList,
    get_route,
    get_route_baseline_roadblock_linkedlist,
    get_route_simplified,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center
from shapely import Point

from asim.simulation.gym.environment.simulation_wrapper import SimulationWrapper


@dataclass
class ProgressCache:

    expert_route_roadblocks: Optional[RouteRoadBlockLinkedList]
    expert_progress: float  # [m]
    progress_computer: Optional[PerFrameProgressAlongRouteComputer] = None


def calculate_route_completion_human(ego_states: List[EgoState], scenario_simulation: SimulationWrapper) -> float:
    """
    Calculates route completion relative to the human trajectory from the logs. Used as default.
    :param ego_states: List of ego states from the simulation.
    :param scenario_simulation: Simulation wrapper object containing the scenario simulation.
    :return: Route completion delta, which is the difference in route completion from the last time step to the current one (normalized).
    """
    assert len(ego_states) > 2
    current_ego_state = ego_states[-1]
    ego_linestring = scenario_simulation.ego_linestring
    current_route_completion = ego_linestring.project(Point(*current_ego_state.center.array), normalized=True)
    past_route_completion = scenario_simulation.route_completion
    route_completion_delta = np.maximum(0.0, current_route_completion - past_route_completion)
    scenario_simulation.update_route_completion(current_route_completion)
    return route_completion_delta


def calculate_route_completion_mean(ego_states: List[EgoState], scenario_simulation: SimulationWrapper) -> float:
    """
    Calculates route completion relative to the overall average in the logs (i.e. 62 meter)
    NOTE: This lead to aggressive ego behavior. Function likely removed in the future.
    :param ego_states: List of ego states from the simulation.
    :param scenario_simulation: Simulation wrapper object containing the scenario simulation.
    :return: Route completion delta, which is the difference in route completion from the last time step to the current one (normalized).
    """
    MEAN_ROUTE_COMPLETION: float = 62.0  # [m]
    ego_linestring = PDMPath([ego_state.center for ego_state in scenario_simulation.simulation_ego_states]).linestring
    current_route_completion = np.clip(ego_linestring.length / MEAN_ROUTE_COMPLETION, 0.0, 1.0)
    past_route_completion = scenario_simulation.route_completion
    route_completion_delta = np.clip(current_route_completion - past_route_completion, 0.0, 1.0)
    scenario_simulation.update_route_completion(current_route_completion)
    return route_completion_delta


def calculate_route_completion_nuplan(
    ego_states: List[EgoState],
    scenario_simulation: SimulationWrapper,
    progress_cache: Optional[ProgressCache] = None,
    score_progress_threshold: float = 0.001,
) -> float:
    """
    Calculates route completion based on the nuPlan progress metric.
    NOTE: This function worked okay, but did not lead to better results compared to the human trajectory.
        The implementation is also more complex. We might remove it in the future.
    :param ego_states: List of ego states from the simulation.
    :param scenario_simulation: Simulation wrapper object containing the scenario simulation.
    :return: Route completion delta, which is the difference in route completion from the last time step to the current one (normalized).
    """

    first_iteration = progress_cache is None

    # 1. Calculate expert route and progress
    if first_iteration:
        scenario = scenario_simulation.scenario
        expert_states = scenario.get_expert_ego_trajectory()
        expert_poses = extract_ego_center(expert_states)

        expert_route = get_route(map_api=scenario.map_api, poses=expert_poses)
        expert_route_simplified = get_route_simplified(expert_route)
        expert_route_roadblocks = get_route_baseline_roadblock_linkedlist(scenario.map_api, expert_route_simplified)

        if expert_route_roadblocks.head is None:
            progress_cache = ProgressCache(expert_route_roadblocks, 0.0)
        else:
            expert_progress_computer = PerFrameProgressAlongRouteComputer(expert_route_roadblocks)
            expert_progress = np.sum(expert_progress_computer(ego_poses=expert_poses))
            ego_progress_computer = PerFrameProgressAlongRouteComputer(expert_route_roadblocks)
            progress_cache = ProgressCache(expert_route_roadblocks, expert_progress, ego_progress_computer)

    # 2. Whether or not valid route was found:
    #   - Return standard values for no progress.
    #   - Calculate new route completion.
    if progress_cache.expert_route_roadblocks.head is None:
        scenario_simulation.update_route_completion(1.0)
        route_completion_delta = 0.0

    else:
        ego_poses = extract_ego_center(ego_states[-2:])
        ego_progress = np.sum(progress_cache.progress_computer(ego_poses=ego_poses))

        current_route_completion = np.clip(
            max(ego_progress, score_progress_threshold) / max(progress_cache.expert_progress, score_progress_threshold),
            0.0,
            1.0,
        )
        past_route_completion = scenario_simulation.route_completion

        route_completion_delta = np.clip(current_route_completion - past_route_completion, 0.0, 1.0)
        scenario_simulation.update_route_completion(current_route_completion)

    return route_completion_delta, progress_cache
