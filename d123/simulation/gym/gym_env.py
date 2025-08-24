from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.common.datatypes.vehicle_state.ego_state import DynamicStateSE2, EgoStateSE2
from d123.dataset.maps.abstract_map import AbstractMap
from d123.dataset.scene.abstract_scene import AbstractScene
from d123.geometry.vector import Vector2D
from d123.simulation.controller.motion_model.kinematic_bicycle_model import KinematicBicycleModel
from d123.simulation.observation.abstract_observation import AbstractObservation
from d123.simulation.observation.log_replay_observation import LogReplayObservation


class GymEnvironment:
    """
    A simple demo environment for testing purposes.
    This class is a placeholder and does not implement any specific functionality.
    """

    def __init__(self, scenes: List[AbstractScene]) -> None:

        self._scenes = scenes
        self._current_iteration = 0
        self._current_scene: Optional[AbstractScene] = None
        self._current_ego_state_se2: Optional[EgoStateSE2] = None

        # self._observation: AbstractObservation = AgentsObservation(None)
        self._observation: AbstractObservation = LogReplayObservation()
        self._observation.initialize()

        self._ego_replay: bool = False

    def reset(self, scene: Optional[AbstractScene]) -> Tuple[AbstractMap, EgoStateSE2, DetectionRecording]:
        """
        Reset the environment to the initial state.
        Returns a tuple containing the map, ego vehicle state, and detection observation.
        """
        if scene is not None:
            self._current_scene = scene
        else:
            self._current_scene = np.random.choice(self._scenes)

        self._current_scene_index = 0
        self._current_ego_state_se2 = self._current_scene.get_ego_state_at_iteration(
            self._current_scene_index
        ).ego_state_se2
        detection_observation = self._observation.reset(self._current_scene)

        return self._current_scene.map_api, self._current_ego_state_se2, detection_observation, self._current_scene

    def step(self, action: npt.NDArray[np.float64]) -> Tuple[EgoStateSE2, DetectionRecording, bool]:
        self._current_scene_index += 1
        if self._ego_replay:
            self._current_ego_state_se2 = self._current_scene.get_ego_state_at_iteration(
                self._current_scene_index
            ).ego_state_se2
        else:
            dynamic_car_state = dynamic_state_from_action(ego_state=self._current_ego_state_se2, action=action)
            next_timepoint = self._current_scene.get_timepoint_at_iteration(self._current_scene_index)
            self._current_ego_state_se2 = KinematicBicycleModel().step(
                self._current_ego_state_se2, dynamic_car_state, next_timepoint
            )

        detection_observation = self._observation.step()
        is_done = self._current_scene_index == self._current_scene.get_number_of_iterations() - 1

        return self._current_ego_state_se2, detection_observation, is_done


def dynamic_state_from_action(ego_state: EgoStateSE2, action: npt.NDArray[np.float64]) -> DynamicStateSE2:
    """
    Convert the action to a dynamic car state.
    """
    # Assuming action is in the form [acceleration, steering_angle]
    long_acceleration = action[0]
    tire_steering_rate = action[1]

    return DynamicStateSE2(
        velocity=ego_state.dynamic_state_se2.velocity,
        acceleration=Vector2D(long_acceleration, 0.0),
        angular_velocity=ego_state.dynamic_state_se2.angular_velocity,
        tire_steering_rate=tire_steering_rate,
        angular_acceleration=0.0,
    )
