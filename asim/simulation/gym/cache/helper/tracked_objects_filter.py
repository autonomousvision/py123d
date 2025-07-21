from dataclasses import dataclass
from typing import List

import numpy as np
from carl_nuplan.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import Agent, StaticObject, TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap


@dataclass
class TrackedObjectsFilter:
    """Configuration for filtering tracked objects by max count per type."""

    # NOTE: filtering the objects is recommended to bound the run time of several components.
    # Extreme cases of >100 vehicles/pedestrians exist in nuPlan.
    max_vehicles: int = 50
    max_pedestrians: int = 50
    max_bicycles: int = 10
    max_static_objects: int = 10
    ignore_off_road_objects: bool = False


def filter_tracked_objects(
    tracked_objects: TrackedObjects,
    ego_state: EgoState,
    map_api: AbstractMap,
    filter: TrackedObjectsFilter = TrackedObjectsFilter(),
) -> TrackedObjects:
    """
    Filter existing tracked objects based on the provided filter configuration.
    Uses max count per type, closest to ego vehicle, and optionally ignores off-road objects.
    :param tracked_objects: TrackedObjects to filter
    :param ego_state: EgoState of the ego vehicle
    :param map_api: map interface of nuPlan.
    :param filter: Filter configuration dataclass, defaults to TrackedObjectsFilter()
    :return: Filtered TrackedObjects containing static objects and agents.
    """
    static_objects = _filter_static_objects(tracked_objects.get_static_objects(), ego_state, map_api, filter)
    agents = _filter_agents(tracked_objects, ego_state, filter)
    return TrackedObjects(static_objects + agents)


def _filter_static_objects(
    static_objects: List[StaticObject],
    ego_state: EgoState,
    map_api: AbstractMap,
    filter: TrackedObjectsFilter,
) -> List[StaticObject]:
    """
    Filter static objects based on the provided filter configuration.
    :param static_objects: list of StaticObject to filter
    :param ego_state: EgoState of the ego vehicle
    :param map_api: map interface of nuPlan.
    :param filter: Filter configuration dataclass, defaults to TrackedObjectsFilter()
    :return: Filtered list of StaticObject containing only those within the specified distance and count.
    """

    if len(static_objects) == 0:
        return []

    if filter.ignore_off_road_objects:
        drivable_area_map = get_drivable_area_map(map_api, ego_state, map_radius=50)
        center_points = np.array([obj.center.point.array for obj in static_objects], dtype=np.float64)
        center_in_map = drivable_area_map.points_in_polygons(center_points).any(axis=0)
        static_objects = [obj for obj, in_map in zip(static_objects, center_in_map) if in_map]

    return _filter_tracked_objects_distance(static_objects, ego_state, filter.max_static_objects)


def _filter_agents(
    tracked_objects: TrackedObjects,
    ego_state: EgoState,
    filter: TrackedObjectsFilter,
) -> List[Agent]:
    """
    Filter all dynamic agents based on the provided filter configuration.
    :param tracked_objects: Complete tracked objects wrapper of nuPlan.
    :param ego_state: EgoState of the ego vehicle.
    :param filter: Filter configuration dataclass, defaults to TrackedObjectsFilter()
    :return: List of Agent objects filtered by type and distance to the ego vehicle.
    """

    filtered_agents: List[Agent] = []
    for agent_type, agent_max_count in {
        TrackedObjectType.VEHICLE: filter.max_vehicles,
        TrackedObjectType.PEDESTRIAN: filter.max_pedestrians,
        TrackedObjectType.BICYCLE: filter.max_bicycles,
    }.items():
        agents = tracked_objects.get_tracked_objects_of_type(agent_type)
        filtered_agents.extend(_filter_tracked_objects_distance(agents, ego_state, agent_max_count))
    return filtered_agents


def _filter_tracked_objects_distance(
    tracked_objects: List[TrackedObject], ego_state: EgoState, max_objects: int
) -> List[TrackedObject]:
    """
    Helper function to filter tracked objects based on their distance to the ego vehicle.
    :param tracked_objects: list of arbitrary TrackedObject (Agent or StaticObject) to filter
    :param ego_state: EgoState of the ego vehicle for location in global coordinates.
    :param max_objects: Maximum number of objects to return.
    :return: List of TrackedObject filtered by distance to the ego vehicle.
    """

    if len(tracked_objects) <= max_objects:
        return tracked_objects

    center_points = np.array([obj.center.point.array for obj in tracked_objects], dtype=np.float64)
    distance_to_ego = np.linalg.norm(center_points - ego_state.center.point.array, axis=1)
    max_nearest_objects_indices = np.argsort(distance_to_ego)[:max_objects]

    return [tracked_objects[idx] for idx in max_nearest_objects_indices]
