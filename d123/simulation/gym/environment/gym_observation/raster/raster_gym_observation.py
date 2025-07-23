from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from d123.simulation.gym.environment.gym_observation.abstract_gym_observation import AbstractGymObservation
from d123.simulation.gym.environment.gym_observation.raster.raster_renderer import RasterRenderer
from d123.simulation.gym.environment.helper.environment_area import AbstractEnvironmentArea
from d123.simulation.gym.environment.helper.environment_cache import (
    BoxDetectionCache,
    MapCache,
    build_environment_caches,
)
from d123.simulation.planning.abstract_planner import PlannerInitialization, PlannerInput


def del_keys_in_dict(info: Dict[str, Any], keys: List[str]) -> None:
    """
    Deletes specified keys from the info dictionary if they exist.
    :param info: Dictionary from which keys will be deleted.
    :param keys: List of keys to delete from the dictionary.
    """
    for key in keys:
        if key in info.keys():
            del info[key]


class RasterObservationType(IntEnum):
    """Enum to represent behavior at different stages in the RasterGymObservation."""

    INFERENCE = 0
    RESET = 1
    STEP = 2


class RasterGymObservation(AbstractGymObservation):
    """Default raster observation builder for the CaRL model."""

    def __init__(
        self,
        environment_area: AbstractEnvironmentArea,
        renderer: RasterRenderer,
        obs_num_measurements: int = 10,
        num_value_measurements: int = 4,
        action_space_dim: int = 2,
        inference: bool = False,
    ) -> None:
        """
        Initializes the RasterGymObservation.
        :param environment_area: Environment area to be used for rendering.
        :param renderer: Renderer class, see implementation of DefaultRenderer.
        :param obs_num_measurements: number of observation measurements passed to the policy, defaults to 10
        :param num_value_measurements: number of value measurements passed to the value network, defaults to 4
        :param action_space_dim: dimension of action space (steering and acceleration), defaults to 2
        :param inference: Whether the observation builder is used during inference, defaults to False
        """

        self._environment_area = environment_area
        self._renderer = renderer

        self._obs_num_measurements = obs_num_measurements
        self._num_value_measurements = num_value_measurements
        self._action_space_dim = action_space_dim

        self._inference = inference

        # lazy loaded during inference
        # NOTE: route roadblocks of current scenario are stored, as they may require correction during inference
        self._route_lane_group_ids: Optional[List[str]] = None

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._route_lane_group_ids = None

    def get_observation_space(self) -> spaces.Space:
        """Inherited, see superclass."""
        return spaces.Dict(
            {
                "bev_semantics": spaces.Box(
                    0,
                    255,
                    shape=self._renderer.shape,
                    dtype=np.uint8,
                ),
                "measurements": spaces.Box(
                    -math.inf,
                    math.inf,
                    shape=(self._obs_num_measurements,),
                    dtype=np.float32,
                ),
                "value_measurements": spaces.Box(
                    -math.inf,
                    math.inf,
                    shape=(self._num_value_measurements,),
                    dtype=np.float32,
                ),
            }
        )

    def get_gym_observation(
        self,
        planner_input: PlannerInput,
        planner_initialization: PlannerInitialization,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inherited, see superclass."""

        if self._inference:
            observation_type = RasterObservationType.INFERENCE
        elif planner_input.iteration.index == 0:
            observation_type = RasterObservationType.RESET
        else:
            observation_type = RasterObservationType.STEP

        observation = {}
        observation["bev_semantics"] = self._get_bev_semantics(
            planner_input,
            planner_initialization,
            observation_type,
            info,
        )
        observation["measurements"] = self._get_build_measurements(planner_input, observation_type, info)
        observation["value_measurements"] = self._get_value_measurements(observation_type, info)

        return observation

    def _get_bev_semantics(
        self,
        planner_input: PlannerInput,
        planner_initialization: PlannerInitialization,
        observation_type: RasterObservationType,
        info: Dict[str, Any],
    ) -> npt.NDArray[np.uint8]:
        """
        Helper function to build BEV raster of the current environment step.
        :param planner_input: Planner input interface of d123.
        :param planner_initialization: Planner initialization interface of d123.
        :param observation_type: Enum whether to render for inference, reset or step.
        :param info: Arbitrary information dictionary, for passing information between modules.
        :raises ValueError: If the DefaultObservationType is invalid.
        :return: BEV raster as a numpy array.
        """
        # FIXME:
        # if observation_type == RasterObservationType.INFERENCE:

        if observation_type in [RasterObservationType.INFERENCE, RasterObservationType.RESET]:
            map_cache, detection_cache = build_environment_caches(
                planner_input, planner_initialization, self._environment_area
            )
        elif observation_type == RasterObservationType.STEP:
            assert "map_cache" in info.keys()
            assert "detection_cache" in info.keys()
            assert isinstance(info["map_cache"], MapCache)
            assert isinstance(info["detection_cache"], BoxDetectionCache)
            map_cache, detection_cache = info["map_cache"], info["detection_cache"]
        else:
            raise ValueError("RasterObservationType is invalid")

        del_keys_in_dict(info, ["map_cache", "detection_cache"])
        return self._renderer.render(map_cache, detection_cache)

    def _get_build_measurements(
        self,
        planner_input: PlannerInput,
        observation_type: RasterObservationType,
        info: Dict[str, Any],
    ) -> npt.NDArray[np.float32]:
        """
        Helper function to build measurements of the current environment step.
        :param planner_input: Planner input interface of d123.
        :param observation_type: Enum whether to render for inference, reset or step.
        :param info: Arbitrary information dictionary, for passing information between modules.
        :return: Ego measurements as a numpy array.
        """

        if "last_action" in info.keys():
            assert observation_type in [
                RasterObservationType.INFERENCE,
                RasterObservationType.STEP,
            ]
            assert len(info["last_action"]) == self._action_space_dim
            last_action = info["last_action"]
        else:
            assert observation_type in [
                RasterObservationType.INFERENCE,
                RasterObservationType.RESET,
            ]
            last_action = np.zeros(self._action_space_dim, dtype=np.float32)

        ego_state = planner_input.history.current_state[0]
        last_acceleration, last_steering_rate = last_action[0], last_action[1]

        # FIXME: rear axle to center conversion of kinematic states
        # state_array = ego_state_to_center_state_array(ego_state)
        # observation_measurements = np.array(
        #     [
        #         last_acceleration,
        #         last_steering_rate,
        #         state_array[StateIndex.VELOCITY_X],
        #         state_array[StateIndex.VELOCITY_Y],
        #         state_array[StateIndex.ACCELERATION_X],
        #         state_array[StateIndex.ACCELERATION_Y],
        #         state_array[StateIndex.STEERING_ANGLE],
        #         state_array[StateIndex.STEERING_RATE],
        #         state_array[StateIndex.ANGULAR_VELOCITY],
        #         state_array[StateIndex.ANGULAR_ACCELERATION],
        #     ],
        #     dtype=np.float32,
        # )
        observation_measurements = np.array(
            [
                last_acceleration,
                last_steering_rate,
                ego_state.dynamic_state_se2.velocity.x,
                ego_state.dynamic_state_se2.velocity.y,
                ego_state.dynamic_state_se2.acceleration.x,
                ego_state.dynamic_state_se2.acceleration.y,
                ego_state.tire_steering_angle,
                ego_state.dynamic_state_se2.tire_steering_rate,
                ego_state.dynamic_state_se2.angular_velocity,
                ego_state.dynamic_state_se2.angular_acceleration,
            ],
            dtype=np.float32,
        )
        del_keys_in_dict(info, ["last_action"])
        return observation_measurements

    def _get_value_measurements(
        self,
        observation_type: RasterObservationType,
        info: Dict[str, Any],
    ) -> npt.NDArray[np.float32]:
        """
        Helper function to build value measurements of the current environment step.
        :param observation_type: Enum whether to render for inference, reset or step.
        :param info: Arbitrary information dictionary, for passing information between modules.
        :raises ValueError: If the DefaultObservationType is invalid.
        :return: Value measurements as a numpy array.
        """

        if observation_type in [
            RasterObservationType.INFERENCE,
            RasterObservationType.RESET,
        ]:
            remaining_time = 1.0
            remaining_progress = 1.0
            comfort_score = 1.0
            ttc_score = 1.0
        elif observation_type == RasterObservationType.STEP:
            assert "remaining_time" in info.keys()
            assert "remaining_progress" in info.keys()
            assert "comfort_score" in info.keys()
            assert "ttc_score" in info.keys()
            remaining_time = info["remaining_time"]
            remaining_progress = info["remaining_progress"]
            comfort_score = info["comfort_score"]
            ttc_score = info["ttc_score"]
        else:
            raise ValueError("DefaultObservationType is invalid")

        value_measurements = np.array(
            [
                remaining_time,
                remaining_progress,
                comfort_score,
                ttc_score,
            ],
            dtype=np.float32,
        )
        del_keys_in_dict(info, ["remaining_time", "remaining_progress", "comfort_score", "ttc_score"])
        return value_measurements
