"""
NOTE @DanielDauner:

This file may be cleaned up in the future.

"""

from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from carl_nuplan.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import states_se2_to_array
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from nuplan.common.actor_state.ego_state import EgoState, StateSE2
from nuplan.common.maps.abstract_map import Lane
from nuplan.common.maps.abstract_map_objects import LaneConnector, LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from shapely import Point, Polygon

from d123.simulation.gym.environment.helper.environment_area import AbstractEnvironmentArea
from d123.simulation.gym.environment.helper.environment_cache import (
    BoxDetectionCache,
    MapCache,
    environment_cache_manager,
)
from d123.simulation.gym.environment.reward_builder.abstract_reward_builder import AbstractRewardBuilder
from d123.simulation.gym.environment.reward_builder.components.collision import (
    calculate_all_collisions,
    calculate_at_fault_collision,
    calculate_non_stationary_collisions,
)
from d123.simulation.gym.environment.reward_builder.components.comfort import (
    calculate_action_delta_comfort,
    calculate_kinematics_comfort,
    calculate_kinematics_comfort_fixed,
    calculate_kinematics_comfort_legacy,
    calculate_kinematics_history_comfort,
)
from d123.simulation.gym.environment.reward_builder.components.off_route import (
    calculate_off_route_v1,
    calculate_off_route_v2,
)
from d123.simulation.gym.environment.reward_builder.components.progress import (
    ProgressCache,
    calculate_route_completion_human,
    calculate_route_completion_mean,
    calculate_route_completion_nuplan,
)
from d123.simulation.gym.environment.reward_builder.components.time_to_collision import (
    FAIL_TTC,
    SUCCESS_TTC,
    calculate_ttc_v1,
    calculate_ttc_v2,
)
from d123.simulation.gym.environment.simulation_wrapper import SimulationWrapper

NUM_SCENARIO_ITERATIONS: Final[int] = 150  # TODO: Remove this constant.


@dataclass
class DefaultRewardComponents:
    """Dataclass to store the components of the default reward builder."""

    route_completion: float = 0.0

    # hard constraints
    blocked: bool = False  # not implemented
    red_light: bool = False
    collision: bool = False
    stop_sign: bool = False  # not implemented
    off_road: bool = False

    # soft constraints
    lane_distance: float = 1.0
    too_fast: float = 1.0
    off_route: float = 1.0
    comfort: float = 1.0
    ttc: float = 1.0

    @property
    def hard_constraints(self) -> List[bool]:
        """
        :return: boolean values of the hard constraints, i.e. collision, that lead to termination.
        """
        return [
            self.blocked,
            self.red_light,
            self.collision,
            self.stop_sign,
            self.off_road,
        ]

    @property
    def soft_constraints(self) -> List[float]:
        """
        :return: float values of the soft constraints.
        """
        return [
            self.lane_distance,
            self.too_fast,
            self.off_route,
            self.comfort,
            self.ttc,
        ]


@dataclass
class DefaultRewardConfig:
    """Configuration for the default reward builder."""

    route_completion_type: Optional[str] = "human"
    collision_type: Optional[str] = "non_stationary"
    ttc_type: Optional[str] = "v2"
    red_light_type: Optional[str] = None
    lane_distance_type: Optional[str] = "v1"
    off_route_type: Optional[str] = "v1"
    comfort_type: Optional[str] = "kinematics"

    comfort_accumulation: str = "value"
    ttc_accumulation: str = "value"
    reward_accumulation: str = "regular"

    terminal_penalty: float = 0.0
    collision_terminal_penalty: float = 0.0
    off_road_violation_threshold: float = 0.0
    lane_distance_violation_threshold: float = 0.5
    survival_ratio: float = 0.6

    reward_factor: float = 100.0

    def __post_init__(self):
        assert self.route_completion_type is None or self.route_completion_type in [
            "human",
            "mean",
            "nuplan",
        ]
        assert self.collision_type is None or self.collision_type in [
            "all",
            "non_stationary",
            "at_fault",
        ]
        assert self.ttc_type is None or self.ttc_type in ["v1", "v2"]
        assert self.red_light_type is None or self.red_light_type in ["v1"]
        assert self.lane_distance_type is None or self.lane_distance_type in ["v1"]
        assert self.off_route_type is None or self.off_route_type in ["v1", "v2"]
        assert self.comfort_type is None or self.comfort_type in [
            "action_delta",
            "kinematics",
            "kinematics_legacy",
            "kinematics_history",
            "kinematics_fixed",
        ]
        assert self.comfort_accumulation in ["terminal", "value"]
        assert self.ttc_accumulation in ["terminal", "value"]
        assert self.reward_accumulation in ["regular", "nuplan", "survival"]


class DefaultRewardBuilder(AbstractRewardBuilder):
    """Default reward builder for the Gym simulation environment."""

    def __init__(self, environment_area: AbstractEnvironmentArea, config: DefaultRewardConfig) -> None:
        """
        Initializes the default reward builder.
        :param environment_area: Environment area class that defines the map patch to calculate the reward.
        :param config: Configuration for the default reward builder.
        """

        self._environment_area = environment_area
        self._config = config

        # lazy loaded
        self._reward_history: List[DefaultRewardComponents] = []
        self._prev_collided_track_tokens: List[str] = []
        self._expert_red_light_infractions: List[str] = []
        self._comfort_values: List[npt.NDArray[np.bool_]] = []
        self._progress_cache: Optional[ProgressCache] = None

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._reward_history: List[DefaultRewardComponents] = []
        self._prev_collided_track_tokens: List[str] = []
        self._expert_red_light_infractions: List[str] = []
        self._comfort_values: List[npt.NDArray[np.bool_]] = []
        self._progress_cache: Optional[ProgressCache] = None

    def build_reward(self, simulation_wrapper: SimulationWrapper, info: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """Inherited, see superclass."""

        map_cache, detection_cache = environment_cache_manager.build_environment_caches(
            planner_input=simulation_wrapper.current_planner_input,
            planner_initialization=simulation_wrapper.planner_initialization,
            environment_area=self._environment_area,
        )
        info["map_cache"] = map_cache
        info["detection_cache"] = detection_cache

        reward_components = self._calculate_reward_components(simulation_wrapper, map_cache, detection_cache)

        if self._config.reward_accumulation == "nuplan":
            reward, termination, truncation = self._nuplan_accumulate(reward_components)
        elif self._config.reward_accumulation == "survival":
            reward, termination, truncation = self._survival_accumulate(reward_components)
        else:
            reward, termination, truncation = self._regular_accumulate(reward_components)
        self._reward_history.append(reward_components)

        if termination or truncation or not simulation_wrapper.is_simulation_running():
            info["reward"] = self._accumulate_info()
            info["comfort"] = self._accumulate_info_comfort()

        self._add_value_measurements(simulation_wrapper, info)
        return reward, termination, truncation

    def _regular_accumulate(self, reward_components: DefaultRewardComponents) -> Tuple[float, bool, bool]:
        """
        Accumulate the reward components into a single reward value, as described in CaRL paper.
        TODO: Refactor this method.
        :param reward_components: Dataclass storing reward components.
        :return:
            - reward: The accumulated reward value.
            - termination: Whether the simulation terminates in the current step.
            - truncation: Whether the simulation is truncated in the current step.
        """
        termination = any(reward_components.hard_constraints)
        if self._config.comfort_accumulation == "terminal":
            termination = termination or reward_components.comfort < 1.0
        if self._config.ttc_accumulation == "terminal":
            termination = termination or reward_components.ttc < 1.0
        truncation = termination
        terminal_penalty = (
            self._config.collision_terminal_penalty if reward_components.collision else self._config.terminal_penalty
        )
        terminate_factor = 0.0 if termination else 1.0
        reward = (
            reward_components.route_completion * np.prod(reward_components.soft_constraints) * terminate_factor
            + terminal_penalty
        )
        return reward * self._config.reward_factor, termination, truncation

    def _survival_accumulate(self, reward_components: DefaultRewardComponents) -> Tuple[float, bool, bool]:
        """
        Accumulate the reward components into a single reward value, and adding a survival bonus.
        TODO: Refactor this method.
        :param reward_components: Dataclass storing reward components.
        :return:
            - reward: The accumulated reward value.
            - termination: Whether the simulation terminates in the current step.
            - truncation: Whether the simulation is truncated in the current step.
        """
        termination = any(reward_components.hard_constraints)
        truncation = termination
        terminal_penalty = (
            self._config.collision_terminal_penalty if reward_components.collision else self._config.terminal_penalty
        )
        terminate_factor = 0.0 if termination else 1.0
        raw_reward = (1 - self._config.survival_ratio) * reward_components.route_completion * np.prod(
            reward_components.soft_constraints
        ) + (self._config.survival_ratio / NUM_SCENARIO_ITERATIONS)

        reward = raw_reward * terminate_factor + terminal_penalty
        return reward * self._config.reward_factor, termination, truncation

    def _nuplan_accumulate(self, reward_components: DefaultRewardComponents) -> Tuple[float, bool, bool]:
        """
        Accumulate the reward components into a single reward value, using a weighted combination similar to nuPlan.
        TODO: Refactor this method.
        :param reward_components: Dataclass storing reward components.
        :return:
            - reward: The accumulated reward value.
            - termination: Whether the simulation terminates in the current step.
            - truncation: Whether the simulation is truncated in the current step.
        """

        termination = any(reward_components.hard_constraints)
        truncation = termination
        reward = 0.0

        if not termination:

            progress = reward_components.route_completion

            ttc = 1.0 if reward_components.ttc == 1.0 else 0.0
            speed = reward_components.too_fast
            comfort = 1.0 if reward_components.comfort == 1.0 else 0.0

            ttc /= NUM_SCENARIO_ITERATIONS
            speed /= NUM_SCENARIO_ITERATIONS
            comfort /= NUM_SCENARIO_ITERATIONS

            reward = (5 * progress + 5 * ttc + 4 * speed + 2 * comfort) / 16
            reward = reward * reward_components.off_route * reward_components.lane_distance * self._config.reward_factor

        return reward, termination, truncation

    def _accumulate_info(self) -> Dict[str, float]:
        """
        Helper function to log the accumulated reward information.
        TODO: Remove this method.
        """
        reward_info: Dict[str, float] = {}
        reward_info["reward_progress"] = np.sum([reward.route_completion for reward in self._reward_history])

        # reward_info["reward_blocked"] = not np.any([reward.blocked for reward in self._reward_history])
        reward_info["reward_red_light"] = not np.any([reward.red_light for reward in self._reward_history])
        reward_info["reward_collision"] = not np.any([reward.collision for reward in self._reward_history])
        # reward_info["reward_stop_sign"] = not np.any([reward.stop_sign for reward in self._reward_history])
        reward_info["reward_off_road"] = not np.any([reward.off_road for reward in self._reward_history])

        reward_info["reward_lane_distance"] = np.mean([reward.lane_distance for reward in self._reward_history])
        reward_info["reward_too_fast"] = np.mean([reward.too_fast for reward in self._reward_history])
        reward_info["reward_off_route"] = np.mean([reward.off_route for reward in self._reward_history])
        reward_info["reward_comfort"] = not np.any([self._reward_history[-1].comfort < 1.0])
        reward_info["reward_ttc"] = not np.any([reward.ttc < 1.0 for reward in self._reward_history])

        for key, value in reward_info.items():
            reward_info[key] = float(value)

        return reward_info

    def _accumulate_info_comfort(self) -> Dict[str, float]:
        """
        Helper function to log the accumulated comfort information.
        TODO: Remove this method.
        """
        comfort_info: Dict[str, float] = {}
        comfort = np.array(self._comfort_values, dtype=np.bool_)

        comfort_info["comfort_lon_acceleration"] = comfort[-1, 0]
        comfort_info["comfort_lat_acceleration"] = comfort[-1, 1]
        comfort_info["comfort_jerk_metric"] = comfort[-1, 2]
        comfort_info["comfort_lon_jerk_metric"] = comfort[-1, 3]
        comfort_info["comfort_yaw_accel"] = comfort[-1, 4]
        comfort_info["comfort_yaw_rate"] = comfort[-1, 5]

        for key, value in comfort_info.items():
            comfort_info[key] = float(value)

        return comfort_info

    def _add_value_measurements(self, simulation_wrapper: SimulationWrapper, info: Dict[str, Any]) -> None:
        """
        Pass some information for the value observation.
        TODO: DEBUG/REMOVE.
        :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
        :param info: Arbitrary information dictionary, for passing information between modules.
        """
        assert len(self._reward_history) > 0

        current_iteration = simulation_wrapper.current_planner_input.iteration.index
        num_simulation_iterations = simulation_wrapper.scenario.get_number_of_iterations()

        remaining_time = 1 - (current_iteration / num_simulation_iterations)
        remaining_progress = 1 - simulation_wrapper.route_completion
        comfort_score = self._reward_history[-1].comfort
        ttc_score = self._reward_history[-1].ttc

        info["remaining_time"] = remaining_time
        info["remaining_progress"] = remaining_progress
        info["comfort_score"] = comfort_score
        info["ttc_score"] = ttc_score

    def _calculate_reward_components(
        self, simulation_wrapper: SimulationWrapper, map_cache: MapCache, detection_cache: BoxDetectionCache
    ) -> DefaultRewardComponents:
        """
        Internal method to calculate the reward components based on the current simulation state.
        :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
        :param map_cache: Cache map elements in the environment area.
        :param detection_cache: Cached objects for detection tracks in the current simulation step.
        :return: dataclass containing the reward components.
        """
        ego_states = simulation_wrapper.current_planner_input.history.ego_states
        current_ego_state = ego_states[-1]

        component_dict: Dict[str, Union[bool, float]] = {}

        current_lane, intersecting_lanes = _find_current_and_intersecting_lanes(current_ego_state, map_cache)

        # ---------------- Route Completion ----------------
        if self._config.route_completion_type is not None:
            if self._config.route_completion_type == "human":
                component_dict["route_completion"] = calculate_route_completion_human(ego_states, simulation_wrapper)
            elif self._config.route_completion_type == "mean":
                component_dict["route_completion"] = calculate_route_completion_mean(ego_states, simulation_wrapper)
            elif self._config.route_completion_type == "nuplan":
                (
                    component_dict["route_completion"],
                    progress_cache,
                ) = calculate_route_completion_nuplan(ego_states, simulation_wrapper, self._progress_cache)
                self._progress_cache = progress_cache
            else:
                raise ValueError(f"Invalid route completion type: {self._config.route_completion_type}")

        # ---------------- Hard Constraints ----------------

        # 5. Off road
        component_dict["off_road"], in_multiple_lanes = _calculate_off_road(
            current_ego_state,
            map_cache,
            intersecting_lanes,
            self._config.off_road_violation_threshold,
        )
        in_multiple_lanes_or_offroad = in_multiple_lanes or component_dict["off_road"]

        # 1. Is ego blocked for 90s
        # component_dict["blocked"] = _calculate_blocked()  # Not implemented

        # 2. Red light infraction
        if self._config.red_light_type is not None:
            if self._config.red_light_type == "v1":
                (
                    component_dict["red_light"],
                    self._expert_red_light_infractions,
                ) = _calculate_red_light(
                    simulation_wrapper,
                    map_cache,
                    current_lane,
                    self._expert_red_light_infractions,
                )
            else:
                raise ValueError(f"Invalid red light type: {self._config.red_light_type}")

        # 3. Collision
        if self._config.collision_type is not None:
            if self._config.collision_type == "all":
                collision, collided_track_tokens = calculate_all_collisions(
                    current_ego_state,
                    detection_cache.tracked_objects,
                    self._prev_collided_track_tokens,
                )
            elif self._config.collision_type == "non_stationary":
                collision, collided_track_tokens = calculate_non_stationary_collisions(
                    current_ego_state,
                    detection_cache.tracked_objects,
                    self._prev_collided_track_tokens,
                )
            elif self._config.collision_type == "at_fault":
                collision, collided_track_tokens = calculate_at_fault_collision(
                    current_ego_state,
                    detection_cache.tracked_objects,
                    self._prev_collided_track_tokens,
                    in_multiple_lanes_or_offroad,
                )
            else:
                raise ValueError(f"Invalid collision type: {self._config.collision_type}")

            component_dict["collision"] = collision
            self._prev_collided_track_tokens.extend(collided_track_tokens)

        # 4. Stop signs
        # component_dict["stop_sign"] = _calculate_stop_sign()  # Not implemented

        # ---------------- Soft Constraints ----------------

        # 1. Lane Distance
        if self._config.lane_distance_type is not None:
            if self._config.lane_distance_type == "v1":
                component_dict["lane_distance"] = _calculate_lane_distance(
                    current_ego_state,
                    current_lane,
                    self._config.lane_distance_violation_threshold,
                )
            else:
                raise ValueError(f"Invalid lane distance type: {self._config.lane_distance_type}")

        # 2. Driving too fast
        component_dict["too_fast"] = _calculate_too_fast(current_ego_state, current_lane)

        # 3. Driving off route
        if self._config.off_route_type is not None:
            if self._config.off_route_type == "v1":
                component_dict["off_route"] = calculate_off_route_v1(simulation_wrapper, map_cache)
            elif self._config.off_route_type == "v2":
                component_dict["off_route"] = calculate_off_route_v2(simulation_wrapper, map_cache)
            else:
                raise ValueError(f"Invalid off route type: {self._config.off_route_type}")

        # 4. comfort
        if self._config.comfort_type is not None:
            comfort_results = None
            if self._config.comfort_type == "action_delta":
                component_dict["comfort"] = calculate_action_delta_comfort(simulation_wrapper)
            elif self._config.comfort_type == "kinematics":
                (
                    component_dict["comfort"],
                    comfort_results,
                ) = calculate_kinematics_comfort(simulation_wrapper)
            elif self._config.comfort_type == "kinematics_legacy":
                (
                    component_dict["comfort"],
                    comfort_results,
                ) = calculate_kinematics_comfort_legacy(simulation_wrapper)
            elif self._config.comfort_type == "kinematics_history":
                (
                    component_dict["comfort"],
                    comfort_results,
                ) = calculate_kinematics_history_comfort(simulation_wrapper)
            elif self._config.comfort_type == "kinematics_fixed":
                (
                    component_dict["comfort"],
                    comfort_results,
                ) = calculate_kinematics_comfort_fixed(simulation_wrapper)
            else:
                raise ValueError(f"Invalid comfort type: {self._config.comfort_type}")

            if comfort_results is not None:
                self._comfort_values.append(comfort_results)

        # 5. Time to collision
        if self._config.ttc_type is not None:
            ttc_failed_previously = any([reward.ttc < SUCCESS_TTC for reward in self._reward_history])
            if ttc_failed_previously:
                component_dict["ttc"] = FAIL_TTC
            elif self._config.ttc_type == "v1":
                component_dict["ttc"] = calculate_ttc_v1(simulation_wrapper)
            elif self._config.ttc_type == "v2":
                component_dict["ttc"] = calculate_ttc_v2(
                    simulation_wrapper,
                    self._prev_collided_track_tokens,
                    in_multiple_lanes_or_offroad,
                )
            else:
                raise ValueError(f"Invalid ttc type: {self._config.ttc_type}")

        return DefaultRewardComponents(**component_dict)


def _calculate_blocked() -> bool:
    """Placeholder for blocked calculation. TODO: remove."""
    return False


def _calculate_red_light(
    simulation_wrapper: SimulationWrapper,
    map_cache: MapCache,
    current_lane: Optional[LaneGraphEdgeMapObject],
    expert_red_light_infractions: List[str],
) -> Tuple[bool, List[str]]:
    """
    Calculates the red light infraction based in the current iteration.
    TODO: Refactor this method.
    :param simulation_wrapper: Wrapper object containing complete nuPlan simulation.
    :param map_cache: Cache map elements in the environment area.
    :param current_lane: Lane object aligned to the ego vehicle in the current iteration.
    :param expert_red_light_infractions: List of traffic light infractions of the human expert.
    :return: Whether the ego vehicle is violating a red light and the updated list of expert red light infractions.
    """

    STOPPED_SPEED_THRESHOLD: float = 5e-02

    iteration = simulation_wrapper.current_planner_input.iteration.index
    ego_state = simulation_wrapper.current_ego_state
    expert_ego_state = simulation_wrapper.scenario.get_ego_state_at_iteration(iteration)

    ego_on_lane = current_lane is None or isinstance(current_lane, Lane)
    ego_stopped = ego_state.dynamic_car_state.speed < STOPPED_SPEED_THRESHOLD

    if ego_on_lane or ego_stopped:
        return False, expert_red_light_infractions

    # add on route checking
    red_connectors: Dict[str, LaneConnector] = {}
    for connector_id, connector in map_cache.lane_connectors.items():
        if (
            (connector.get_roadblock_id() in map_cache.route_lane_group_ids)
            and (connector_id in map_cache.traffic_lights.keys())
            and (map_cache.traffic_lights[connector_id] == TrafficLightStatusType.RED)
        ):
            red_connectors[connector_id] = connector

    red_connector_map = PDMOccupancyMap(
        list(red_connectors.keys()),
        [connector.polygon for connector in red_connectors.values()],
    )
    ego_center_point = Point(*ego_state.center.array)
    expert_center_point = Point(*expert_ego_state.center.array)

    ego_intersecting_connectors = red_connector_map.intersects(ego_center_point)
    expert_intersecting_connectors = red_connector_map.intersects(expert_center_point)
    expert_red_light_infractions.extend(expert_intersecting_connectors)

    non_covered_infractions = list(set(ego_intersecting_connectors) - set(expert_red_light_infractions))
    if len(non_covered_infractions) > 0:
        return True, expert_red_light_infractions

    return False, expert_red_light_infractions


def _calculate_stop_sign() -> bool:
    """Placeholder for stop sign infraction. TODO: remove."""
    return False


def _calculate_off_road(
    ego_state: EgoState,
    map_cache: MapCache,
    intersecting_lanes: List[LaneGraphEdgeMapObject],
    violation_threshold: float,
) -> Tuple[bool, bool]:
    """
    Calculates whether the ego vehicle is off-road based on its corners and the drivable area map.
    :param ego_state: Ego vehicle state of the current iteration.
    :param map_cache: Cache map elements in the environment area.
    :param intersecting_lanes: List of lanes that intersect with the ego vehicle's position.
    :param violation_threshold: Threshold distance to consider a corner as off-road.
    :return: Whether the ego vehicle is off-road and whether it is in multiple lanes.
    """

    drivable_area_map = map_cache.drivable_area_map
    ego_corners = np.array(
        [[point.x, point.y] for point in ego_state.agent.box.all_corners()],
        dtype=np.float64,
    )
    corner_in_polygons = drivable_area_map.contains_vectorized(ego_corners)  # (geom, 4)

    polygon_indices = np.where(corner_in_polygons.sum(axis=-1) > 0)[0]
    corners_dwithin_polygons = corner_in_polygons.sum(axis=0) > 0

    if violation_threshold > 0.0 and not np.all(corners_dwithin_polygons):
        ego_polygons = [drivable_area_map.geometries[idx] for idx in polygon_indices]
        ego_polygons.extend([lane.polygon for lane in intersecting_lanes])

        for corner_idx in np.where(~corners_dwithin_polygons)[0]:
            distances = [polygon.distance(Point(*ego_corners[corner_idx])) for polygon in ego_polygons]
            if len(distances) > 0 and min(distances) < violation_threshold:
                corners_dwithin_polygons[corner_idx] = True

    off_road = not np.all(corners_dwithin_polygons)
    in_multiple_lanes = len(polygon_indices) > 1

    return off_road, in_multiple_lanes


def _find_current_and_intersecting_lanes(
    ego_state: EgoState,
    map_cache: MapCache,
) -> Tuple[Optional[LaneGraphEdgeMapObject], List[LaneGraphEdgeMapObject]]:
    """
    Helper function to find the current lane and intersecting lanes based on the ego vehicle's state.
    TODO: Refactor this method.
    :param ego_state: Ego vehicle state of the current iteration.
    :param map_cache: Cache map elements in the environment area.
    :return: Tuple of
        - current_lane: The lane that the ego vehicle is currently on, or None if not found.
        - intersecting_lanes: List of lanes that intersect with the ego vehicle's position.
    """

    current_lane: Optional[LaneGraphEdgeMapObject] = None

    # store lanes and lane connectors in common dict
    lanes_dict: Dict[str, LaneGraphEdgeMapObject] = {
        lane.id: lane for lane in list(map_cache.lanes.values()) + list(map_cache.lane_connectors.values())
    }

    # find intersecting lanes
    lane_polygons: List[Polygon] = [lane.polygon for lane in lanes_dict.values()]
    lane_map = PDMOccupancyMap(list(lanes_dict.keys()), lane_polygons)
    ego_center_point = Point(ego_state.center.x, ego_state.center.y)
    intersecting_lanes_ids = lane_map.intersects(ego_center_point)
    intersecting_lanes = [lanes_dict[lane_id] for lane_id in intersecting_lanes_ids]

    def _calculate_heading_error(ego_pose: StateSE2, lane: LaneGraphEdgeMapObject) -> float:
        """Calculate the heading error of the ego vehicle with respect to the lane."""

        # calculate nearest state on baseline
        lane_se2_array = states_se2_to_array(lane.baseline_path.discrete_path)
        lane_distances = np.linalg.norm(ego_pose.point.array[None, ...] - lane_se2_array[..., :2], axis=-1)

        # calculate heading error
        heading_error = lane.baseline_path.discrete_path[np.argmin(lane_distances)].heading - ego_pose.heading
        heading_error = np.abs(normalize_angle(heading_error))

        return heading_error

    if len(intersecting_lanes_ids) > 0:
        lane_route_errors: Dict[str, float] = {}
        lane_errors: Dict[str, float] = {}

        for lane_id in intersecting_lanes_ids:
            lane = lanes_dict[lane_id]
            heading_error = _calculate_heading_error(ego_state.center, lane)
            if lane.get_roadblock_id() in map_cache.route_lane_group_ids:
                lane_route_errors[lane_id] = heading_error
            lane_errors[lane_id] = heading_error

        # Search for lanes on route first
        if len(lane_route_errors) > 0:
            current_lane = lanes_dict[min(lane_route_errors, key=lane_route_errors.get)]

        else:  # Fallback to all intersecting lanes
            current_lane = lanes_dict[min(lane_errors, key=lane_errors.get)]

    return current_lane, intersecting_lanes


def _calculate_lane_distance(
    ego_state: EgoState,
    current_lane: Optional[LaneGraphEdgeMapObject],
    lane_distance_violation_threshold: float = 0.5,
) -> float:
    """
    Calculates the distance of the ego vehicle to the center of the current lane.
    Normalizes the distance to a value between 0 and 1.
    :param ego_state: Ego vehicle state of the current iteration.
    :param current_lane: Lane object aligned to the ego vehicle in the current iteration.
    :param Normed distance, for which ego is not penalized: _description_, defaults to 0.5
    :return: Normed reward distance between 0 and 1, where 1 is the best value.
    """

    if current_lane is not None and isinstance(current_lane, Lane):
        ego_center_point = Point(ego_state.center.x, ego_state.center.y)
        center_distance, left_distance, right_distance = (
            ego_center_point.distance(current_lane.baseline_path.linestring),
            ego_center_point.distance(current_lane.left_boundary.linestring),
            ego_center_point.distance(current_lane.right_boundary.linestring),
        )

        # assumes that the ego center is in lane polygon
        center_distance_norm = (center_distance - lane_distance_violation_threshold) / (
            (center_distance + np.minimum(left_distance, right_distance) - lane_distance_violation_threshold) + 1e-6
        )
        center_distance_norm = np.clip(center_distance_norm, 0, 1)
        return 1.0 - (center_distance_norm * 0.5)

    return 1.0


def _calculate_too_fast(
    ego_state: EgoState,
    current_lane: Optional[LaneGraphEdgeMapObject],
    max_overspeed_value_threshold: float = 2.23,
) -> float:
    """
    Calculates the speed of the ego vehicle in relation to the speed limit of the current lane.
    :param ego_state: Ego vehicle state of the current iteration.
    :param current_lane: Lane object aligned to the ego vehicle in the current iteration.
    :param max_overspeed_value_threshold: max exceeding value for linear penalty, defaults to 2.23
    :return: Reward value between 0 and 1, where 1 is the best value.
    """

    # Adding a small tolerance to handle cases where max_overspeed_value_threshold is specified as 0
    max_overspeed_value_threshold_ = max(max_overspeed_value_threshold, 1e-3)

    if current_lane is not None:
        speed_limit = current_lane.speed_limit_mps
        if speed_limit is not None:
            exceeding_speed = ego_state.dynamic_car_state.speed - speed_limit
            if exceeding_speed > 0.0:
                violation_loss = exceeding_speed / max_overspeed_value_threshold_
                return float(max(0.0, 1.0 - violation_loss))
    return 1.0
