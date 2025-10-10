import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import CAP_STYLE, Polygon

from d123.common.datatypes.detection.detection import BoxDetection, BoxDetectionSE2
from d123.datasets.maps.abstract_map import AbstractMap
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.datatypes.scene.arrow.utils.arrow_getters import BoxDetectionWrapper
from d123.geometry.bounding_box import BoundingBoxSE2
from d123.geometry.point import Point2D
from d123.geometry.polyline import PolylineSE2
from d123.geometry.se import StateSE2
from d123.geometry.transform.tranform_2d import translate_along_yaw
from d123.geometry.vector import Vector2D
from d123.simulation.agents.abstract_agents import AbstractAgents


@dataclass
class IDMConfig:
    target_velocity: float = 10.0  # [m/s]
    min_gap_to_lead_agent: float = 1.0  # [m]
    headway_time: float = 1.5  # [s]
    accel_max: float = 1.0  # [m/s^2]
    decel_max: float = 2.0  # [m/s^2]
    acceleration_exponent: float = 4.0  # Usually set to 4


class IDMAgents(AbstractAgents):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    def __init__(self) -> None:
        """
        Initialize the constant velocity agents.
        """
        super().__init__()
        self._timestep_s: float = 0.1
        self._current_iteration: int = 0
        self._map_api: AbstractMap = None

        self._idm_config: IDMConfig = IDMConfig(target_velocity=5.0, accel_max=1.0, decel_max=2.0)

        self._initial_target_agents: List[BoxDetection] = []
        self._past_target_agents: List[BoxDetection] = []
        self._agent_paths: Dict[str, PolylineSE2] = {}
        self._agent_paths_buffer: Dict[str, Polygon] = {}
        self._agent_initial_vel: Dict[str, float] = {}
        self._extend_path_length: float = 100

    @abstractmethod
    def reset(
        self,
        map_api: AbstractMap,
        target_agents: List[BoxDetection],
        non_target_agents: List[BoxDetection],
        scene: Optional[AbstractScene] = None,
    ) -> List[BoxDetection]:
        assert scene is not None

        self._map_api = map_api
        self._current_iteration = 0
        self._initial_target_agents = [copy.deepcopy(agent) for agent in target_agents]

        future_box_detections = [
            scene.get_box_detections_at_iteration(iteration) for iteration in range(0, scene.number_of_iterations)
        ]

        # TODO: refactor or move for general use
        for agent in self._initial_target_agents:
            future_trajectory: List[StateSE2] = []
            for box_detections in future_box_detections:
                agent_at_iteration = box_detections.get_detection_by_track_token(agent.metadata.track_token)
                if agent_at_iteration is None:
                    break

                future_trajectory.append(agent_at_iteration.center.state_se2)

            if len(future_trajectory) < 2:
                future_trajectory = [agent.center.state_se2, translate_along_yaw(agent.center, Point2D(0.1, 0.0))]

            future_trajectory.append(translate_along_yaw(future_trajectory[-1], Point2D(self._extend_path_length, 0.0)))

            polyline_se2 = PolylineSE2.from_discrete_se2(future_trajectory)
            self._agent_paths[agent.metadata.track_token] = polyline_se2
            self._agent_paths_buffer[agent.metadata.track_token] = polyline_se2.linestring.buffer(
                agent.bounding_box_se2.width / 2, cap_style=CAP_STYLE.square
            )
            self._agent_initial_vel[agent.metadata.track_token] = float(agent.velocity.vector_2d.magnitude)

        self._past_target_agents = self._initial_target_agents
        return self._initial_target_agents

    def step(self, non_target_agents: List[BoxDetection]):
        self._current_iteration += 1

        box_detections = BoxDetectionWrapper(box_detections=non_target_agents + self._past_target_agents)
        occupancy_map = box_detections.occupancy_map

        # time_delta_s = self._timestep_s * self._current_iteration
        current_target_agents = []
        for past_agent in self._past_target_agents:
            agent_velocity: float = float(past_agent.velocity.vector_2d.magnitude)

            agent_path = self._agent_paths[past_agent.metadata.track_token]
            agent_path_buffer = self._agent_paths_buffer[past_agent.metadata.track_token]
            agent_distance_on_path = agent_path.project(past_agent.center.point_2d)

            track_token_in_path: List[str] = occupancy_map.intersects(agent_path_buffer)

            leading_agent: Optional[BoxDetection] = None
            leading_agent_distance_on_path: float = float("inf")
            for track_token in track_token_in_path:
                if track_token == past_agent.metadata.track_token:
                    continue

                other_agent = box_detections.get_detection_by_track_token(track_token)
                if other_agent is None:
                    continue

                other_agent_distance_on_path = agent_path.project(other_agent.center.point_2d)
                if other_agent_distance_on_path < agent_distance_on_path:
                    continue

                if other_agent_distance_on_path < leading_agent_distance_on_path:
                    leading_agent = other_agent
                    leading_agent_distance_on_path = other_agent_distance_on_path

            if leading_agent is not None:
                distance_to_lead_agent = past_agent.shapely_polygon.distance(leading_agent.shapely_polygon)
                lead_agent_velocity = float(leading_agent.velocity.vector_2d.magnitude)
            else:
                distance_to_lead_agent = float(
                    np.clip(agent_path.length - agent_distance_on_path, a_min=0.0, a_max=None)
                )
                lead_agent_velocity = 0.0

            # propagate the agent using IDM
            self._idm_config.target_velocity = self._agent_initial_vel[past_agent.metadata.track_token] + 0.01
            x_dot, v_agent_dot = _propagate_idm(
                agent_velocity, lead_agent_velocity, distance_to_lead_agent, self._idm_config
            )

            v_agent_dot = min(max(-self._idm_config.decel_max, v_agent_dot), self._idm_config.accel_max)
            propagate_distance = agent_distance_on_path + x_dot * self._timestep_s
            propagated_center = agent_path.interpolate(propagate_distance)
            propagated_bounding_box = BoundingBoxSE2(
                propagated_center,
                past_agent.bounding_box_se2.length,
                past_agent.bounding_box_se2.width,
            )
            new_velocity = Vector2D(agent_velocity + v_agent_dot * self._timestep_s, 0.0)
            propagated_agent: BoxDetectionSE2 = BoxDetectionSE2(
                metadata=past_agent.metadata,
                bounding_box_se2=propagated_bounding_box,
                velocity=new_velocity,
            )
            current_target_agents.append(propagated_agent)

        self._past_target_agents = current_target_agents
        return current_target_agents


def _propagate_idm(
    agent_velocity: float, lead_velocity: float, agent_lead_distance: float, idm_config: IDMConfig
) -> Tuple[float, float]:

    # convenience definitions
    s_star = (
        idm_config.min_gap_to_lead_agent
        + agent_velocity * idm_config.headway_time
        + (agent_velocity * (agent_velocity - lead_velocity))
        / (2 * np.sqrt(idm_config.accel_max * idm_config.decel_max))
    )
    s_alpha = max(agent_lead_distance, idm_config.min_gap_to_lead_agent)  # clamp to avoid zero division

    # differential equations
    x_dot = agent_velocity
    try:
        v_agent_dot = idm_config.accel_max * (
            1
            - (agent_velocity / idm_config.target_velocity) ** idm_config.acceleration_exponent
            - (s_star / s_alpha) ** 2
        )
    except:  # noqa: E722
        print("input", agent_velocity, lead_velocity, agent_lead_distance)
        print("s_star", s_star)
        print("s_alpha", s_alpha)
        print("x_dot", x_dot)
        v_agent_dot = 0.0
    return [x_dot, v_agent_dot]
