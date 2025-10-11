from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.geometry.compute import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel

from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.conversion.maps.abstract_map import AbstractMap
from d123.conversion.scene.abstract_scene import AbstractScene
from d123.datatypes.scene.arrow.utils.arrow_getters import EgoStateSE3
from d123.simulation.observation.abstract_observation import AbstractObservation
from d123.simulation.observation.agents_observation import AgentsObservation

# from d123.simulation.observation.log_replay_observation import LogReplayObservation


class DemoGymEnv:
    """
    A simple demo environment for testing purposes.
    This class is a placeholder and does not implement any specific functionality.
    """

    def __init__(self, scenes: List[AbstractScene]) -> None:

        self._scenes = scenes
        self._current_iteration = 0
        self._current_scene: Optional[AbstractScene] = None
        self._current_ego_vehicle_state: Optional[EgoState] = None

        self._observation: AbstractObservation = AgentsObservation(None)
        # self._observation: AbstractObservation = LogReplayObservation()
        self._observation.initialize()

        self._ego_replay: bool = True

    def reset(self, scene: Optional[AbstractScene]) -> Tuple[AbstractMap, EgoState, DetectionRecording]:
        """
        Reset the environment to the initial state.
        Returns a tuple containing the map, ego vehicle state, and detection observation.
        """
        if scene is not None:
            self._current_scene = scene
        else:
            self._current_scene = np.random.choice(self._scenes)

        self._current_scene_index = 0
        self._current_ego_vehicle_state = to_nuplan_ego_vehicle_state(
            self._current_scene.get_ego_vehicle_state_at_iteration(self._current_scene_index)
        )
        # detection_observation = DetectionRecording(
        #     box_detections=self._current_scene.get_box_detections_at_iteration(self._current_scene_index),
        #     traffic_light_detections=self._current_scene.get_traffic_light_detections_at_iteration(
        #         self._current_scene_index
        #     ),
        # )
        detection_observation = self._observation.reset(self._current_scene)

        return (
            self._current_scene.get_map_api(),
            self._current_ego_vehicle_state,
            detection_observation,
            self._current_scene,
        )

    def step(self, action: npt.NDArray[np.float64]) -> Tuple[EgoState, DetectionRecording, bool]:
        self._current_scene_index += 1
        if self._ego_replay:
            ego_vehicle_state = self._current_scene.get_ego_vehicle_state_at_iteration(self._current_scene_index)
            self._current_ego_vehicle_state = to_nuplan_ego_vehicle_state(ego_vehicle_state)
        else:
            dynamic_car_state = get_dynamic_car_state(ego_state=self._current_ego_vehicle_state, action=action)
            self._current_ego_vehicle_state = KinematicBicycleModel(get_pacifica_parameters()).propagate_state(
                self._current_ego_vehicle_state, dynamic_car_state, TimePoint(int(0.1 * int(1e6)))
            )

        detection_observation = self._observation.step()
        is_done = self._current_scene_index == self._current_scene.number_of_iterations - 1

        return self._current_ego_vehicle_state, detection_observation, is_done


def to_nuplan_ego_vehicle_state(ego_vehicle_state: EgoStateSE3) -> EgoState:
    """
    Convert a custom EgoVehicleState to a NuPlan EgoVehicleState.
    This is a placeholder function and should be implemented based on the actual structure of EgoVehicleState.
    """

    # Assuming EgoVehicleState has attributes like position, velocity, heading, etc.
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(
            ego_vehicle_state.bounding_box.center.x,
            ego_vehicle_state.bounding_box.center.y,
            ego_vehicle_state.bounding_box.center.yaw,
        ),
        rear_axle_velocity_2d=StateVector2D(
            ego_vehicle_state.dynamic_state.velocity.x, ego_vehicle_state.dynamic_state.velocity.y
        ),
        rear_axle_acceleration_2d=StateVector2D(
            ego_vehicle_state.dynamic_state.acceleration.x, ego_vehicle_state.dynamic_state.acceleration.y
        ),
        tire_steering_angle=0.0,
        time_point=TimePoint(0),
        vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True,
        angular_vel=ego_vehicle_state.dynamic_state.angular_velocity.z,
        angular_accel=0.0,
        tire_steering_rate=0.0,
    )


def get_dynamic_car_state(ego_state: EgoState, action: npt.NDArray[np.float64]) -> DynamicCarState:
    acceleration, steering_rate = action[0], action[1]
    return DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=ego_state.car_footprint.rear_axle_to_center_dist,
        rear_axle_velocity_2d=ego_state.dynamic_car_state.rear_axle_velocity_2d,
        rear_axle_acceleration_2d=StateVector2D(acceleration, 0),
        tire_steering_rate=steering_rate,
    )
