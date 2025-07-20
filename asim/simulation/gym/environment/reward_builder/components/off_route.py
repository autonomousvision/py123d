from typing import Dict

import numpy as np
from carl_nuplan.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from shapely import Polygon

from asim.simulation.gym.environment.helper.environment_cache import MapCache
from asim.simulation.gym.environment.simulation_wrapper import SimulationWrapper


def calculate_off_route_v1(simulation_wrapper: SimulationWrapper, map_cache: MapCache) -> float:
    """
    Calculate the off-route based on the polygons of the route roadblocks and roadblock connectors polygons.
    NOTE: The route roadblock connector polygons often have a strange shape. We expect the ego learned to exploit this
        during strange overtaking maneuvers. We fixed this in the v2 version below.
    TODO: Refactor or remove this implementation.
    :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
    :param map_cache: Map cache object storing relevant nearby map objects.
    :return: 1.0 if the ego is on route, 0.0 if the ego is off route.
    """
    iteration = simulation_wrapper.current_planner_input.iteration.index

    ego_state = simulation_wrapper.current_ego_state
    expert_ego_state = simulation_wrapper.scenario.get_ego_state_at_iteration(iteration)

    route_roadblocks: Dict[str, Polygon] = {}
    for route_roadblock_id in map_cache.route_lane_group_ids:
        if route_roadblock_id in map_cache.roadblocks:
            route_roadblocks[route_roadblock_id] = map_cache.roadblocks[route_roadblock_id].polygon
        if route_roadblock_id in map_cache.roadblock_connectors:
            route_roadblocks[route_roadblock_id] = map_cache.roadblock_connectors[route_roadblock_id].polygon

    route_map = PDMOccupancyMap(list(route_roadblocks.keys()), list(route_roadblocks.values()))

    points = np.array(
        [
            [point.x, point.y]
            for point in [
                ego_state.center.point,
                expert_ego_state.center.point,
            ]
        ],
        dtype=np.float64,
    )
    points_in_polygon = route_map.points_in_polygons(points)
    on_route = points_in_polygon.sum(axis=0) > 0
    ego_on_route, expert_on_route = on_route[0], on_route[1]

    if not ego_on_route and expert_on_route:
        return 0.0

    return 1.0


def calculate_off_route_v2(simulation_wrapper: SimulationWrapper, map_cache: MapCache) -> float:
    """
    Calculate the off-route based on the polygons of the route roadblocks and roadblock connectors polygons.
    NOTE: This implementation uses the lane/lane-connector polygons instead of the roadblock/roadblock-connector polygons.
    :param simulation_wrapper: Complete simulation wrapper object used for gym simulation.
    :param map_cache: Map cache object storing relevant nearby map objects.
    :return: 1.0 if the ego is on route, 0.0 if the ego is off route.
    """
    iteration = simulation_wrapper.current_planner_input.iteration.index

    ego_state = simulation_wrapper.current_ego_state
    expert_ego_state = simulation_wrapper.scenario.get_ego_state_at_iteration(iteration)

    route_lane_polygons: Dict[str, Polygon] = {}

    for lane_dict in [map_cache.lanes, map_cache.lane_connectors]:
        lane_dict: Dict[str, LaneGraphEdgeMapObject]
        for lane_id, lane in lane_dict.items():
            if lane.get_roadblock_id() in map_cache.route_lane_group_ids:
                route_lane_polygons[lane_id] = lane.polygon

    route_map = PDMOccupancyMap(list(route_lane_polygons.keys()), list(route_lane_polygons.values()))

    points = np.array(
        [
            [point.x, point.y]
            for point in [
                ego_state.center.point,
                expert_ego_state.center.point,
            ]
        ],
        dtype=np.float64,
    )
    points_in_polygon = route_map.points_in_polygons(points)
    on_route = points_in_polygon.sum(axis=0) > 0
    ego_on_route, expert_on_route = on_route[0], on_route[1]

    if not ego_on_route and expert_on_route:
        return 0.0

    return 1.0
