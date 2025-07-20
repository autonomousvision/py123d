import os
from typing import Any, Generator, List, Optional, Set, Tuple, Type

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatuses, Transform
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, SensorChannel, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from asim.simulation.gym.cache.gym_scenario_data import GymScenarioData, MapEnum
from asim.simulation.gym.cache.helper.convert_gym_scenario import (
    deserialize_ego_state,
    deserialize_tracked_objects,
    deserialize_traffic_lights,
)

NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT")


class GymScenario(AbstractScenario):
    """
    A class representing a cached scenario.
    This class is backend-agnostic, and serves as a interface to precomputed features.
    NOTE: This class does not implement all methods of AbstractScenario.
    TODO: Extend this class to implement all methods of AbstractScenario.
    """

    def __init__(self, token: str, scenario_type: str, log_name: str, data: GymScenarioData) -> None:
        """
        Initializes the GymScenario object.
        :param token: Unique identifier for the scenario.
        :param scenario_type: Type of the scenario, according to nuPlan scenario types.
        :param log_name: Name of the log this scenario belongs to.
        :param data: GymScenarioData object containing the scenario data.
        """
        self._token = token
        self._scenario_type = scenario_type
        self._log_name = log_name
        self._data = data

        self._map_api = get_maps_api(
            map_root=NUPLAN_MAPS_ROOT,
            map_version="nuplan-maps-v1.0",
            map_name=MapEnum(data.map).fullname,
        )

    def __reduce__(self) -> Tuple[Type["GymScenario"], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (
            self.__class__,
            (
                self._token,
                self._log_name,
                self._scenario_type,
                self._data,
            ),
        )

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self._token

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return get_pacifica_parameters()

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> str:
        """Inherited, see superclass."""
        return self._map_api

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return 0.1

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return len(self._data.time_points)

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations()
        return TimePoint(self._data.time_points[iteration])

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        raise NotImplementedError("GymScenario does not implement get_lidar_to_ego_transform.")

    def get_mission_goal(self) -> StateSE2:
        """Inherited, see superclass."""
        return StateSE2(0, 0, 0)

    def get_route_lane_group_ids(self) -> List[str]:
        """Inherited, see superclass."""
        return self._data.route_lane_group_ids

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        raise NotImplementedError("GymScenario does not implement get_expert_goal_state.")

    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations()
        return DetectionsTracks(
            deserialize_tracked_objects(
                self._data.tracked_objects[iteration],
                self.get_time_point(iteration),
            )
        )

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        raise NotImplementedError("GymScenario does not implement get_tracked_objects_within_time_window_at_iteration.")

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        """Inherited, see superclass."""
        raise NotImplementedError("GymScenario does not implement get_sensors_at_iteration.")

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations()
        return deserialize_ego_state(
            self._data.ego_states[iteration],
            self.get_time_point(iteration),
            self.ego_vehicle_parameters,
        )

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations()
        return deserialize_traffic_lights(self._data.traffic_lights[iteration], self.get_time_point(iteration))

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """Inherited, see superclass."""
        yield self.get_traffic_light_status_at_iteration(0)

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """Inherited, see superclass."""
        stop_iteration = int(time_horizon / self.database_interval)
        iterations = np.arange(iteration, min(self.get_number_of_iterations(), stop_iteration))
        for future_iteration in iterations:
            yield self.get_traffic_light_status_at_iteration(future_iteration)

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        stop_iteration = int(time_horizon / self.database_interval)
        iterations = np.arange(iteration, min(self.get_number_of_iterations(), stop_iteration))
        for future_iteration in iterations:
            yield self.get_time_point(future_iteration)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        yield self.get_time_point(0)

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        stop_iteration = int(time_horizon / self.database_interval)
        iterations = np.arange(iteration, min(self.get_number_of_iterations(), stop_iteration))
        for future_iteration in iterations:
            yield self.get_ego_state_at_iteration(future_iteration)

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""

        for iteration in range(len(self._data.past_ego_states)):
            yield deserialize_ego_state(
                self._data.past_ego_states[iteration],
                TimePoint(self._data.past_time_points[iteration]),
                self.ego_vehicle_parameters,
            )

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("GymScenario does not implement get_past_sensors.")

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        # TODO: Refactor
        for iteration in range(len(self._data.past_ego_states)):
            yield self.get_tracked_objects_at_iteration(0)

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        stop_iteration = int(time_horizon / self.database_interval)
        iterations = np.arange(iteration, min(self.get_number_of_iterations(), stop_iteration))
        for future_iteration in iterations:
            yield self.get_tracked_objects_at_iteration(future_iteration)
