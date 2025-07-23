from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from carl_nuplan.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState, StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import (
    _get_ego_tracks_displacement_info,
    _get_relevant_tracks,
)
from nuplan.planning.simulation.observation.idm.utils import (
    is_agent_ahead,
    is_agent_behind,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from shapely import creation

from d123.simulation.gym.environment.simulation_wrapper import SimulationWrapper

# TODO: Add to config.
STOPPED_SPEED_THRESHOLD: Final[float] = 5e-03  # [m/s] (ttc)
SUCCESS_TTC: Final[float] = 1.0
FAIL_TTC: Final[float] = 0.5


def _get_coords_array(oriented_box: OrientedBox) -> npt.NDArray[np.float64]:
    """
    Helper function to get corner coordinates of an oriented box.
    :param oriented_box: OrientedBox object from nuPlan.
    :return: numpy array with shape (5, 2) containing the closed corner coordinates of the oriented box.
    """
    coords_array = np.zeros((len(BBCoordsIndex), 2), dtype=np.float64)
    corners = oriented_box.all_corners()
    coords_array[BBCoordsIndex.FRONT_LEFT] = corners[0].array
    coords_array[BBCoordsIndex.REAR_LEFT] = corners[1].array
    coords_array[BBCoordsIndex.REAR_RIGHT] = corners[2].array
    coords_array[BBCoordsIndex.FRONT_RIGHT] = corners[3].array
    coords_array[BBCoordsIndex.CENTER] = corners[0].array  # close polygon
    return coords_array


def _get_dxy(heading: float, velocity: float) -> npt.NDArray[np.float64]:
    """
    Get the displacement vector (global x,y) to propagate a bounding box along its heading with a given velocity.
    :param heading: Heading angle of the bounding box [rad].
    :param velocity: Velocity of the bounding box [m/s].
    :return: Displacement vector (x, y) to propagate the bounding box.
    """
    dxy = np.stack(
        [
            np.cos(heading) * velocity,
            np.sin(heading) * velocity,
        ],
        axis=-1,
    )
    return dxy


def calculate_ttc_v1(simulation_wrapper: SimulationWrapper, resolution: int = 2) -> float:
    """
    Calculate the time to collision (TTC) of the ego vehicle, based on the constant velocity forecast.
    NOTE: Uses less complex logic than the v2 version. TTC required many steps to converge. We are unsure if v2 was required.
    TODO: Refactor or remove this implementation.
    :param simulation_wrapper: Simulation wrapper object containing the scenario simulation.
    :param resolution: The temporal resolution to check collisions (i.e. every two steps), defaults to 2
    :return: 1.0 if no collision is expected, 0.5 if a collision is expected.
    """

    future_time_indices = np.arange(1, 10, resolution, dtype=int)
    future_time_deltas = future_time_indices * simulation_wrapper.scenario.database_interval

    (
        ego_state,
        observation,
    ) = simulation_wrapper.current_planner_input.history.current_state
    assert isinstance(observation, DetectionsTracks)
    tracked_objects = observation.tracked_objects
    ego_speed = ego_state.dynamic_car_state.center_velocity_2d.magnitude()
    if len(tracked_objects) == 0 or ego_speed < STOPPED_SPEED_THRESHOLD:
        return SUCCESS_TTC

    unique_tracked_objects: Dict[str, TrackedObject] = {
        tracked_object.track_token: tracked_object for tracked_object in tracked_objects
    }
    map_api = simulation_wrapper.scenario.map_api
    ego_in_intersection = map_api.is_in_layer(ego_state.rear_axle, layer=SemanticMapLayer.INTERSECTION)

    def _add_object(tracked_object: TrackedObject) -> bool:
        if is_agent_ahead(ego_state.rear_axle, tracked_object.center) or (
            ego_in_intersection and not is_agent_behind(ego_state.rear_axle, tracked_object.center)
        ):
            return True
        return False

    # extract static object polygons
    static_object_tokens, static_object_coords_list = [], []
    for static_object in tracked_objects.get_static_objects():
        if _add_object(static_object):
            static_object_tokens.append(static_object.track_token)
            static_object_coords_list.append(_get_coords_array(static_object.box))
    static_object_coords_array = np.array(static_object_coords_list, dtype=np.float64)  # (num_agents, 5, 2)
    if len(static_object_tokens) == 0:
        static_object_polygons = np.array([], dtype=np.object_)
    else:
        static_object_polygons = creation.polygons(static_object_coords_array)

    # extract agents
    agent_tokens, agent_coords_list, agent_dxy = [], [], []
    for agent in tracked_objects.get_agents():
        if _add_object(agent):
            agent_tokens.append(agent.track_token)
            agent_coords_list.append(_get_coords_array(agent.box))
            agent_dxy.append(_get_dxy(agent.box.center.heading, agent.velocity.magnitude()))
    agent_coords_array = np.array(agent_coords_list, dtype=np.float64)  # (num_agents, 5, 2)
    agent_dxy = np.array(agent_dxy, dtype=np.float64)  # (num_agents, 2)
    if len(agent_tokens) == 0:
        projected_agent_polygons = np.array([], dtype=np.object_)

    # extract ego
    ego_coords_array = _get_coords_array(ego_state.car_footprint.oriented_box)  # (5, 2)
    ego_dxy = _get_dxy(ego_state.center.heading, ego_speed)
    ego_displacements = future_time_deltas[:, None, None] * ego_dxy  # (num_steps, 1, 2)
    projected_ego_coords = ego_coords_array[None, ...] + ego_displacements  # (num_steps, 5, 2)
    projected_ego_polygons = creation.polygons(projected_ego_coords)

    for time_delta, ego_polygon in zip(future_time_deltas, projected_ego_polygons):

        # project agents
        if len(agent_tokens) > 0:
            agent_displacements = agent_dxy * time_delta
            projected_agent_coords = agent_coords_array + agent_displacements[:, None, :]
            projected_agent_polygons = creation.polygons(projected_agent_coords)

        polygons = np.concatenate([static_object_polygons, projected_agent_polygons], axis=0)
        occupancy_map = PDMOccupancyMap(tokens=static_object_tokens + agent_tokens, geometries=polygons)

        # check for collisions
        ego_collision = occupancy_map.intersects(ego_polygon)
        if len(ego_collision) > 0:
            for ego_collision_token in ego_collision:
                track_state = unique_tracked_objects[ego_collision_token].center
                if is_agent_ahead(ego_state.rear_axle, track_state) or (
                    (map_api.is_in_layer(ego_state.rear_axle, layer=SemanticMapLayer.INTERSECTION))
                    and not is_agent_behind(ego_state.rear_axle, track_state)
                ):
                    return FAIL_TTC

    return SUCCESS_TTC


def calculate_ttc_v2(
    simulation_wrapper: SimulationWrapper,
    collided_track_tokens: List[str],
    in_multiple_lanes_or_offroad: bool = False,
    resolution: int = 2,
) -> float:
    """
    Calculate the time to collision (TTC) of the ego vehicle, based on the constant velocity forecast.
    NOTE: Uses TTC logic closely aligned to nuPlan's implementation, i.e. first extract relevant tracks, then compute TTC.
    :param simulation_wrapper: Simulation wrapper object containing the scenario simulation.
    :param collided_track_tokens: Detection track tokens that are already collided (ignored).
    :param in_multiple_lanes_or_offroad: Whether the ego agent is in multiple lanes or offroad, defaults to False
    :param resolution: The temporal resolution to check collisions (TODO: remove or implement).
    :return: 1.0 if no collision is expected, 0.5 if a collision is expected.
    """
    (
        ego_state,
        observation,
    ) = simulation_wrapper.current_planner_input.history.current_state
    assert isinstance(observation, DetectionsTracks)
    tracked_objects = observation.tracked_objects

    # Early non-violation conditions
    if len(tracked_objects) == 0 or ego_state.dynamic_car_state.speed <= STOPPED_SPEED_THRESHOLD:
        return SUCCESS_TTC

    (
        tracks_poses,
        tracks_speed,
        tracks_boxes,
    ) = _extract_tracks_info_excluding_collided_tracks(
        ego_state,
        simulation_wrapper.scenario.map_api,
        tracked_objects,
        collided_track_tokens,
        in_multiple_lanes_or_offroad,
    )
    tracks_poses = np.array(tracks_poses, dtype=np.float64)
    tracks_speed = np.array(tracks_speed, dtype=np.float64)
    tracks_boxes = np.array(tracks_boxes)

    ttc_at_index = _compute_time_to_collision_at_timestamp(ego_state, tracks_poses, tracks_speed, tracks_boxes)
    if ttc_at_index is None:
        return SUCCESS_TTC

    return FAIL_TTC


def _extract_tracks_info_excluding_collided_tracks(
    ego_state: EgoState,
    map_api: AbstractMap,
    tracked_objects: TrackedObjects,
    collided_track_tokens: List[str],
    in_multiple_lanes_or_offroad: bool = False,
) -> Tuple[List[List[float]], List[float], List[OrientedBox]]:

    ego_in_intersection = map_api.is_in_layer(ego_state.rear_axle, layer=SemanticMapLayer.INTERSECTION)

    relevant_tracked_objects: List[TrackedObject] = []
    for tracked_object in tracked_objects:
        tracked_object: TrackedObject
        if tracked_object.track_token not in collided_track_tokens:
            if is_agent_ahead(ego_state.rear_axle, tracked_object.center) or (
                (in_multiple_lanes_or_offroad or ego_in_intersection)
                and not is_agent_behind(ego_state.rear_axle, tracked_object.center)
            ):
                relevant_tracked_objects.append(tracked_object)

    tracks_poses: List[List[float]] = [[*tracked_object.center] for tracked_object in tracked_objects]
    tracks_speed: List[float] = [
        tracked_object.velocity.magnitude() if isinstance(tracked_object, Agent) else 0
        for tracked_object in tracked_objects
    ]
    tracks_boxes: List[OrientedBox] = [tracked_object.box for tracked_object in tracked_objects]
    return tracks_poses, tracks_speed, tracks_boxes


def _compute_time_to_collision_at_timestamp(
    ego_state: EgoState,
    tracks_poses: npt.NDArray[np.float64],
    tracks_speed: npt.NDArray[np.float64],
    tracks_boxes: List[OrientedBox],
    time_step_start: float = 0.1,
    time_step_size: float = 0.2,
    time_horizon: float = 1.0,
) -> Optional[float]:

    ego_speed = ego_state.dynamic_car_state.speed

    # Remain default if we don't have any agents or ego is stopped
    if len(tracks_poses) == 0 or ego_speed <= STOPPED_SPEED_THRESHOLD:
        return None

    displacement_info = _get_ego_tracks_displacement_info(
        ego_state, ego_speed, tracks_poses, tracks_speed, time_step_size
    )
    relevant_tracks_mask = _get_relevant_tracks(
        displacement_info.ego_pose,
        displacement_info.ego_box,
        displacement_info.ego_dx,
        displacement_info.ego_dy,
        tracks_poses,
        tracks_boxes,
        displacement_info.tracks_dxy,
        time_step_size,
        time_horizon,
    )

    # If there is no relevant track affecting TTC, remain default
    if not len(relevant_tracks_mask):
        return None

    # Find TTC for relevant tracks by projecting ego and tracks boxes with time_step_size
    for time_to_collision in np.arange(time_step_start, time_horizon, time_step_size):
        # project ego's center pose and footprint with a fixed speed
        displacement_info.ego_pose[:2] += (
            displacement_info.ego_dx,
            displacement_info.ego_dy,
        )
        projected_ego_box = OrientedBox.from_new_pose(
            displacement_info.ego_box, StateSE2(*(displacement_info.ego_pose))
        )
        # project tracks's center pose and footprint with a fixed speed
        tracks_poses[:, :2] += displacement_info.tracks_dxy
        for track_box, track_pose in zip(tracks_boxes[relevant_tracks_mask], tracks_poses[relevant_tracks_mask]):
            projected_track_box = OrientedBox.from_new_pose(track_box, StateSE2(*track_pose))
            if in_collision(projected_ego_box, projected_track_box):
                return float(time_to_collision)

    return None
