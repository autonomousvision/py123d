from typing import Generator, List

import numpy as np
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
    state_array_to_ego_state,
)
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction_v2
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import TrafficLightStatusData

from asim.simulation.gym.cache.gym_scenario_data import (
    GymScenarioData,
    GymTrackedObjects,
    GymTrafficLights,
    MapEnum,
    TrackedObjectIndex,
)
from asim.simulation.gym.cache.helper.tracked_objects_filter import filter_tracked_objects


def extract_to_gym_scenario_data(scenario: AbstractScenario) -> GymScenarioData:
    """
    Extracts the gym scenario cache data from a nuPlan scenario.
    :param scenario: Any class inheriting from AbstractScenario.
    :return: scenario dataclass used in gym simulation.
    """

    # TODO: add to config
    time_horizon: float = 1.0
    num_samples: int = 10

    # 1. metadata
    map = int(MapEnum.from_fullname(scenario.map_api.map_name))

    # 2. data
    num_iterations = scenario.get_number_of_iterations()
    assert num_iterations > 1

    # 2.1 time points
    time_points: List[int] = []
    for iteration in range(num_iterations):
        time_points.append(scenario.get_time_point(iteration).time_us)

    # 2.2 past time points
    past_time_points: List[int] = []
    for past_time_point in scenario.get_past_timestamps(0, time_horizon, num_samples):
        past_time_points.append(past_time_point.time_us)

    # 2.3 route lane group ids
    route_lane_group_ids = route_roadblock_correction_v2(
        scenario.get_ego_state_at_iteration(0),
        scenario.map_api,
        scenario.get_route_lane_group_ids(),
    )

    # 2.4 ego states
    ego_states: List[List[float]] = []
    for iteration in range(num_iterations):
        ego_state = scenario.get_ego_state_at_iteration(iteration)
        ego_states.append(serialize_ego_state(ego_state))

    # 2.5 past ego states
    past_ego_states: List[List[float]] = []
    for past_ego_state in scenario.get_ego_past_trajectory(0, time_horizon, num_samples):
        past_ego_states.append(serialize_ego_state(past_ego_state))

    # 2.6 bounding boxes
    tracked_objects: List[GymTrackedObjects] = []
    for iteration in range(num_iterations):
        tracked_objects_ = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
        tracked_objects_ = filter_tracked_objects(
            tracked_objects_,
            scenario.get_ego_state_at_iteration(iteration),
            scenario.map_api,
        )
        tracked_objects.append(serialize_tracked_objects(tracked_objects_))

    # 2.7 traffic lights
    traffic_lights: List[GymTrafficLights] = []
    for iteration in range(num_iterations):
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(iteration))
        traffic_lights.append(serialize_traffic_lights(traffic_light_data))

    return GymScenarioData(
        map=map,
        time_points=time_points,
        past_time_points=past_time_points,
        route_lane_group_ids=route_lane_group_ids,
        ego_states=ego_states,
        past_ego_states=past_ego_states,
        tracked_objects=tracked_objects,
        traffic_lights=traffic_lights,
    )


def serialize_ego_state(ego_state: EgoState) -> List[float]:
    """
    Extracts ego state information as floats.
    :param ego_state: Ego state object in nuPlan.
    :return: List of floats, see function ego_state_to_state_array
    """
    state_array = ego_state_to_state_array(ego_state)
    return list(state_array[:7])


def serialize_tracked_objects(tracked_objects: TrackedObjects) -> GymTrackedObjects:
    """
    Extract the tracked objects information as GymTrackedObjects dataclass.
    :param tracked_objects: Wrapped object for detection tracks in nuPlan.
    :return: Dataclass containing lightweight tracked objects information.
    """

    states: List[List[float]] = []
    tracked_object_types: List[int] = []
    # tokens: List[str] = []
    track_tokens: List[str] = []

    for tracked_object in tracked_objects:
        tracked_object: TrackedObject

        state = [
            tracked_object.center.x,
            tracked_object.center.y,
            tracked_object.center.heading,
            tracked_object.box.length,
            tracked_object.box.width,
        ]
        if tracked_object.tracked_object_type in AGENT_TYPES:
            state += [tracked_object.velocity.x, tracked_object.velocity.y]
        else:
            state += [0.0, 0.0]

        states.append(state)
        tracked_object_types.append(int(tracked_object.tracked_object_type))
        # tokens.append(tracked_object.token)
        track_tokens.append(tracked_object.track_token)

    # return GymTrackedObjects(states, tracked_object_types, tokens, track_tokens)
    return GymTrackedObjects(states, tracked_object_types, track_tokens)


def serialize_traffic_lights(
    traffic_light_data: List[TrafficLightStatusData],
) -> GymTrafficLights:
    """
    Extract the traffic light information as GymTrafficLights dataclass.
    :param traffic_light_data: List of traffic light status data from nuPlan.
    :return: Dataclass containing traffic light states and lane connector ids.
    """
    states: List[str] = []
    lane_connector_ids: List[str] = []

    for traffic_light_status in traffic_light_data:
        states.append(int(traffic_light_status.status))
        lane_connector_ids.append(traffic_light_status.lane_connector_id)

    return GymTrafficLights(states, lane_connector_ids)


def deserialize_ego_state(
    ego_state: List[float], time_point: TimePoint, vehicle_parameters: VehicleParameters
) -> EgoState:
    """
    Helper function to convert a serialized ego state back to an EgoState object.
    :param ego_state: List of floats representing the ego state.
    :param time_point: TimePoint object representing the time of the state.
    :param vehicle_parameters: VehicleParameters object of ego vehicle.
    :return: EgoState object reconstructed from the list of floats.
    """
    state_array = np.zeros(StateIndex.size(), dtype=np.float64)
    state_array[:7] = ego_state
    return state_array_to_ego_state(state_array, time_point, vehicle_parameters)


def deserialize_tracked_objects(gym_tracked_objects: GymTrackedObjects, time_point: TimePoint) -> TrackedObjects:
    """
    Helper function to convert GymTrackedObjects to nuPlans detection tracks wrapper (TrackedObjects).
    :param gym_tracked_objects: dataclass of tracked object cache.
    :param time_point: Time point of the tracked objects.
    :return: nuPlans detection tracks wrapper (TrackedObjects)
    """

    states = gym_tracked_objects.states
    tracked_object_types = gym_tracked_objects.tracked_object_types
    track_tokens = gym_tracked_objects.track_tokens

    tracked_objects: List[TrackedObject] = []

    assert len(set([len(states), len(tracked_object_types), len(track_tokens)])) == 1

    for state, tracked_object_type, track_token in zip(states, tracked_object_types, track_tokens):
        assert len(state) == TrackedObjectIndex.size()

        oriented_box = OrientedBox(
            center=StateSE2(*state[TrackedObjectIndex.STATE_SE2]),
            length=state[TrackedObjectIndex.LENGTH],
            width=state[TrackedObjectIndex.WIDTH],
            height=1.0,  # dummy
        )

        track_metadata = SceneObjectMetadata(
            timestamp_us=time_point.time_us,
            token=track_token,  # dummy
            track_id=None,  # dummy
            track_token=track_token,
        )

        track_type = TrackedObjectType(tracked_object_type)
        if track_type in AGENT_TYPES:
            velocity = StateVector2D(*state[TrackedObjectIndex.VELOCITY])
            tracked_object = Agent(
                tracked_object_type=track_type,
                oriented_box=oriented_box,
                velocity=velocity,
                metadata=track_metadata,
            )
        else:
            tracked_object = StaticObject(
                tracked_object_type=track_type,
                oriented_box=oriented_box,
                metadata=track_metadata,
            )

        tracked_objects.append(tracked_object)

    return TrackedObjects(tracked_objects)


def deserialize_traffic_lights(
    traffic_lights: GymTrafficLights, time_point: TimePoint
) -> Generator[TrafficLightStatusData, None, None]:
    """
    Helper function to convert GymTrafficLights to nuPlan traffic light status data.
    :param traffic_lights: GymTrafficLights dataclass of cache.
    :param time_point: TimePoint object representing the time of the traffic light status.
    :yield: Generator yielding TrafficLightStatusData objects.
    """
    for state, lane_connector_id in zip(traffic_lights.states, traffic_lights.lane_connector_ids):
        yield TrafficLightStatusData(
            status=TrafficLightStatusType(state),
            lane_connector_id=lane_connector_id,
            timestamp=time_point.time_us,
        )
