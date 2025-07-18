from typing import Dict, Final, List, Tuple

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    _get_collision_type,
)
from nuplan.planning.metrics.utils.collision_utils import CollisionType

STOPPED_SPEED_THRESHOLD: Final[float] = 5e-02  # Threshold of ego considered stationary [mps]


def _get_collisions(
    ego_state: EgoState,
    tracked_objects: TrackedObjects,
    ignore_track_tokens: List[str] = [],
) -> Dict[str, TrackedObject]:
    """
    Helper function to get all tracked objects that ego collides with.
    Ignores agents with tokens in ignore_track_tokens (similar to the nuPlan collision function)
    :param ego_state: Ego state object of nuPlan.
    :param tracked_objects: Tracked object wrapper of nuPlan.
    :param ignore_track_tokens: list of tokens (str) ego collided before, defaults to []
    :return: dictionary of tokens and tracked objects.
    """
    collided_track_dict: Dict[str, TrackedObject] = {}
    for tracked_object in tracked_objects:
        tracked_object: TrackedObject
        if (tracked_object.track_token not in ignore_track_tokens) and in_collision(
            ego_state.car_footprint.oriented_box, tracked_object.box
        ):
            collided_track_dict[tracked_object.track_token] = tracked_object
    return collided_track_dict


def calculate_all_collisions(
    ego_state: EgoState,
    tracked_objects: TrackedObjects,
    prev_collided_track_tokens: List[str] = [],
) -> Tuple[bool, List[str]]:
    """
    Reward term for ego collision. Considers all collision types.
    :param ego_state: Ego state object of nuPlan.
    :param tracked_objects: Tracked object wrapper of nuPlan.
    :param prev_collided_track_tokens: list of tokens (str) ego collided before, defaults to []
    :return: whether ego collides and corresponding detection tokens.
    """
    collided_track_dict = _get_collisions(ego_state, tracked_objects, prev_collided_track_tokens)
    collided_track_tokens = list(collided_track_dict.keys())
    return len(collided_track_tokens) > 0, collided_track_tokens


def calculate_non_stationary_collisions(
    ego_state: EgoState,
    tracked_objects: TrackedObjects,
    prev_collided_track_tokens: List[str],
) -> Tuple[bool, List[str]]:
    """
    Reward term for ego collision. Ignores collision when ego stationary.
    :param ego_state: Ego state object of nuPlan.
    :param tracked_objects: Tracked object wrapper of nuPlan.
    :param prev_collided_track_tokens: list of tokens (str) ego collided before, defaults to []
    :return: whether ego collides and corresponding detection tokens.
    """
    collided_track_dict = _get_collisions(ego_state, tracked_objects, prev_collided_track_tokens)
    collided_track_tokens = list(collided_track_dict.keys())
    ego_stationary = ego_state.dynamic_car_state.speed < STOPPED_SPEED_THRESHOLD
    return (len(collided_track_tokens) > 0 and not ego_stationary), collided_track_tokens


def calculate_at_fault_collision(
    ego_state: EgoState,
    tracked_objects: TrackedObjects,
    prev_collided_track_tokens: List[str],
    in_multiple_lanes_or_offroad: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Reward term for ego collision. Ignores non-at-fault collisions.
    https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/no_ego_at_fault_collisions.py
    :param ego_state: Ego state object of nuPlan.
    :param tracked_objects: Tracked object wrapper of nuPlan.
    :param prev_collided_track_tokens: list of tokens (str) ego collided before, defaults to []
    :param in_multiple_lanes_or_offroad: whether ego is in multiple roads of off road.
    :return: whether ego collides and corresponding detection tokens.
    """

    collided_track_dict = _get_collisions(ego_state, tracked_objects, prev_collided_track_tokens)
    collided_track_tokens = list(collided_track_dict.keys())

    at_fault_collision: bool = False
    for tracked_object in collided_track_dict.values():
        collision_type = _get_collision_type(ego_state, tracked_object)
        collisions_at_stopped_track_or_active_front: bool = collision_type in [
            CollisionType.ACTIVE_FRONT_COLLISION,
            CollisionType.STOPPED_TRACK_COLLISION,
        ]
        collision_at_lateral: bool = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
        if collisions_at_stopped_track_or_active_front or (in_multiple_lanes_or_offroad and collision_at_lateral):
            at_fault_collision = True
            break

    return at_fault_collision, collided_track_tokens
