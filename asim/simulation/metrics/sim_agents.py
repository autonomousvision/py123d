from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from asim.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index
from asim.dataset.arrow.conversion import BoxDetection, DetectionType
from asim.dataset.recording.detection.detection import BoxDetectionWrapper
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.metrics.histogram_metric import BinaryHistogramIntersectionMetric, HistogramIntersectionMetric
from asim.simulation.metrics.interaction_based import _get_collision_feature, _get_object_distance_feature
from asim.simulation.metrics.kinematics import (
    _get_linear_acceleration_from_agents_array,
    _get_linear_speed_from_agents_array,
    _get_yaw_acceleration_from_agents_array,
    _get_yaw_rate_from_agents_array,
)
from asim.simulation.metrics.map_based import _get_offroad_feature, _get_road_center_distance_feature


def get_sim_agents_metrics(scene: AbstractScene, agent_rollouts: List[BoxDetectionWrapper]) -> Dict[str, float]:
    def get_agent_tokens(agent_rollout: List[BoxDetection]) -> List[str]:
        return [
            box_detection.metadata.track_token
            for box_detection in agent_rollout
            if box_detection.metadata.detection_type == DetectionType.VEHICLE
        ]

    # def _get_time_s_from_scene(scene: AbstractScene) -> List[float]:
    #     initial_timepoint = scene.get_timepoint_at_iteration(0)
    #     constant = 0.1
    #     time_s: List[float] = []
    #     for iteration in range(scene.get_number_of_iterations()):
    #         timestep = scene.get_timepoint_at_iteration(iteration)
    #         time_delta = timestep - initial_timepoint
    #         time_s.append(time_delta.time_s + constant)
    #     return time_s

    log_rollouts = [
        scene.get_box_detections_at_iteration(iteration) for iteration in range(scene.get_number_of_iterations())
    ]
    initial_agent_tokens = get_agent_tokens(agent_rollouts[0])
    # time_s = _get_time_s_from_scene(scene)

    log_agents_array, log_agents_mask = _get_log_agents_array(scene, initial_agent_tokens)
    agents_array, agents_mask = _get_rollout_agents_array(agent_rollouts, initial_agent_tokens)

    results: Dict[str, float] = {}

    # 1. Kinematics metrics

    # 1.1 Speed
    speed_metric = HistogramIntersectionMetric(min_val=0.0, max_val=25.0, n_bins=10, name="speed", weight=0.05)
    log_speed = _get_linear_speed_from_agents_array(log_agents_array, log_agents_mask)
    agents_speed = _get_linear_speed_from_agents_array(agents_array, log_agents_mask)
    speed_result = speed_metric.calculate_intersection(log_speed, agents_speed, log_agents_mask)
    results.update(speed_result)

    # 1.2 Acceleration
    acceleration_metric = HistogramIntersectionMetric(
        min_val=-12.0, max_val=12.0, n_bins=11, name="acceleration", weight=0.1
    )
    log_acceleration = _get_linear_acceleration_from_agents_array(log_agents_array, log_agents_mask)
    agents_acceleration = _get_linear_acceleration_from_agents_array(agents_array, log_agents_mask)
    acceleration_result = acceleration_metric.calculate_intersection(
        log_acceleration, agents_acceleration, log_agents_mask
    )
    results.update(acceleration_result)

    # 1.3 Yaw rate
    yaw_rate_metric = HistogramIntersectionMetric(
        min_val=-0.628, max_val=0.628, n_bins=11, name="yaw_rate", weight=0.05
    )
    log_yaw_rate = _get_yaw_rate_from_agents_array(log_agents_array, log_agents_mask)
    agents_yaw_rate = _get_yaw_rate_from_agents_array(agents_array, log_agents_mask)
    yaw_rate_result = yaw_rate_metric.calculate_intersection(log_yaw_rate, agents_yaw_rate, log_agents_mask)
    results.update(yaw_rate_result)

    # 1.4 Yaw acceleration
    yaw_acceleration_metric = HistogramIntersectionMetric(
        min_val=-3.14, max_val=3.14, n_bins=11, name="yaw_acceleration", weight=0.05
    )
    log_yaw_acceleration = _get_yaw_acceleration_from_agents_array(log_agents_array, log_agents_mask)
    agents_yaw_acceleration = _get_yaw_acceleration_from_agents_array(agents_array, log_agents_mask)
    yaw_acceleration_result = yaw_acceleration_metric.calculate_intersection(
        log_yaw_acceleration, agents_yaw_acceleration, log_agents_mask
    )
    results.update(yaw_acceleration_result)

    # 2. Interaction based
    # 2.1 Collision
    collision_metric = BinaryHistogramIntersectionMetric(name="collision", weight=0.25)
    logs_collision = _get_collision_feature(log_agents_array, log_rollouts)
    agents_collision = _get_collision_feature(agents_array, agent_rollouts)
    collision_results = collision_metric.calculate_intersection(agents_collision, logs_collision, log_agents_mask)
    results.update(collision_results)
    # collision_metric.plot_histograms(logs_collision, agents_collision, log_agents_mask)

    # 2.2 TTC
    # TODO: Implement TTC metric

    # 2.3 Object distance
    object_distance_metric = HistogramIntersectionMetric(
        min_val=0.0, max_val=40.0, n_bins=10, name="object_distance", weight=0.15
    )
    agents_object_distance = _get_object_distance_feature(agents_array, agent_rollouts)
    log_object_distance = _get_object_distance_feature(log_agents_array, log_rollouts)
    object_distance_results = object_distance_metric.calculate_intersection(
        log_object_distance, agents_object_distance, log_agents_mask
    )
    results.update(object_distance_results)

    # 3. Map based
    # 3.1 Offroad
    offroad_metric = BinaryHistogramIntersectionMetric(name="offroad", weight=0.25)
    agents_offroad = _get_offroad_feature(agents_array, log_agents_mask, scene.map_api)
    log_offroad = _get_offroad_feature(log_agents_array, log_agents_mask, scene.map_api)
    offroad_results = offroad_metric.calculate_intersection(log_offroad, agents_offroad, log_agents_mask)
    results.update(offroad_results)

    # 3.2 lane center distance
    center_distance_metric = HistogramIntersectionMetric(
        min_val=0.0, max_val=10.0, n_bins=10, name="center_distance", weight=0.15
    )
    log_center_distance = _get_road_center_distance_feature(log_agents_array, log_agents_mask, scene.map_api)
    agent_center_distance = _get_road_center_distance_feature(agents_array, log_agents_mask, scene.map_api)
    center_distance_results = center_distance_metric.calculate_intersection(
        log_center_distance, agent_center_distance, log_agents_mask
    )
    results.update(center_distance_results)

    results["meta_score"] = sum([score for name, score in results.items() if name.endswith("_score")])

    return results


def _get_log_agents_array(
    scene: AbstractScene, agent_tokens: List[str]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    log_agents_array = np.zeros(
        (len(agent_tokens), scene.get_number_of_iterations(), len(BoundingBoxSE2Index)),
        dtype=np.float64,
    )
    log_agents_mask = np.zeros(
        (len(agent_tokens), scene.get_number_of_iterations()),
        dtype=bool,
    )

    for iteration in range(scene.get_number_of_iterations()):
        box_detections = scene.get_box_detections_at_iteration(iteration)
        for agent_idx, agent_token in enumerate(agent_tokens):
            box_detection = box_detections.get_detection_by_track_token(agent_token)
            if box_detection is not None:
                log_agents_mask[agent_idx, iteration] = True
                log_agents_array[agent_idx, iteration] = box_detection.bounding_box_se2.array

    return log_agents_array, log_agents_mask


def _get_rollout_agents_array(
    agent_rollouts: List[BoxDetectionWrapper], agent_tokens: List[str]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rollout_agents_array = np.zeros(
        (len(agent_tokens), len(agent_rollouts), len(BoundingBoxSE2Index)),
        dtype=np.float64,
    )
    rollout_agents_mask = np.zeros(
        (len(agent_tokens), len(agent_rollouts)),
        dtype=bool,
    )

    for iteration, agent_rollout in enumerate(agent_rollouts):
        for agent_idx, agent_token in enumerate(agent_tokens):
            box_detection = agent_rollout.get_detection_by_track_token(agent_token)
            if box_detection is not None:
                rollout_agents_mask[agent_idx, iteration] = True
                rollout_agents_array[agent_idx, iteration] = box_detection.bounding_box_se2.array

    return rollout_agents_array, rollout_agents_mask
