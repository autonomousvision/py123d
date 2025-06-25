from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from asim.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index
from asim.dataset.arrow.conversion import BoxDetection, DetectionType
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.recording.detection.detection import BoxDetectionWrapper
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.metrics.sim_agents.histogram_metric import (
    BinaryHistogramIntersectionMetric,
    HistogramIntersectionMetric,
)
from asim.simulation.metrics.sim_agents.interaction_based import _get_collision_feature, _get_object_distance_feature
from asim.simulation.metrics.sim_agents.kinematics import (
    _get_linear_acceleration_from_agents_array,
    _get_linear_speed_from_agents_array,
    _get_yaw_acceleration_from_agents_array,
    _get_yaw_rate_from_agents_array,
)
from asim.simulation.metrics.sim_agents.map_based import _get_offroad_feature, _get_road_center_distance_feature
from asim.simulation.metrics.sim_agents.utils import _get_log_agents_array, _get_rollout_agents_array


@dataclass
class SimAgentsData:

    mask: npt.NDArray[np.bool]

    # 1. Kinematics
    speed: npt.NDArray[np.float64]
    acceleration: npt.NDArray[np.float64]
    yaw_rate: npt.NDArray[np.float64]
    yaw_acceleration: npt.NDArray[np.float64]

    # 2. Interaction based
    collision: npt.NDArray[np.bool]
    object_distance: npt.NDArray[np.float64]

    # 3. Map based
    offroad: npt.NDArray[np.bool]
    center_distance: npt.NDArray[np.float64]


def get_sim_agents_metrics(scene: AbstractScene, agent_rollouts: List[BoxDetectionWrapper]) -> Dict[str, float]:
    def get_agent_tokens(agent_rollout: List[BoxDetection]) -> List[str]:
        return [
            box_detection.metadata.track_token
            for box_detection in agent_rollout
            if box_detection.metadata.detection_type == DetectionType.VEHICLE
        ]

    log_rollouts = [
        scene.get_box_detections_at_iteration(iteration) for iteration in range(scene.get_number_of_iterations())
    ]
    initial_agent_tokens = get_agent_tokens(agent_rollouts[0])
    log_agents_array, log_agents_mask = _get_log_agents_array(scene, initial_agent_tokens)
    agents_array, agents_mask = _get_rollout_agents_array(agent_rollouts, initial_agent_tokens)

    log_agents_data = _extract_sim_agent_data(log_agents_array, log_agents_mask, log_rollouts, scene.map_api)
    agents_data = _extract_sim_agent_data(agents_array, agents_mask, agent_rollouts, scene.map_api)

    results: Dict[str, float] = {}

    # 0. Other data
    results.update(_collision_rate(log_agents_data, agents_data))
    results.update(_offroad_rate(log_agents_data, agents_data))

    # 1. Kinematics metrics
    # 1.1 Speed
    speed_metric = HistogramIntersectionMetric(min_val=0.0, max_val=25.0, n_bins=10, name="speed", weight=0.05)
    speed_result = speed_metric.compute(log_agents_data.speed, agents_data.speed, log_agents_data.mask)
    results.update(speed_result)

    # 1.2 Acceleration
    acceleration_metric = HistogramIntersectionMetric(
        min_val=-12.0, max_val=12.0, n_bins=11, name="acceleration", weight=0.05
    )
    acceleration_result = acceleration_metric.compute(
        log_agents_data.acceleration,
        agents_data.acceleration,
        log_agents_data.mask,
    )
    results.update(acceleration_result)

    # 1.3 Yaw rate
    yaw_rate_metric = HistogramIntersectionMetric(
        min_val=-0.628, max_val=0.628, n_bins=11, name="yaw_rate", weight=0.05
    )
    yaw_rate_result = yaw_rate_metric.compute(
        log_agents_data.yaw_rate,
        agents_data.yaw_rate,
        log_agents_data.mask,
    )
    results.update(yaw_rate_result)

    # 1.4 Yaw acceleration
    yaw_acceleration_metric = HistogramIntersectionMetric(
        min_val=-3.14, max_val=3.14, n_bins=11, name="yaw_acceleration", weight=0.05
    )
    yaw_acceleration_result = yaw_acceleration_metric.compute(
        log_agents_data.yaw_acceleration,
        agents_data.yaw_acceleration,
        log_agents_data.mask,
    )
    results.update(yaw_acceleration_result)

    # 2. Interaction based
    # 2.1 Collision
    collision_metric = BinaryHistogramIntersectionMetric(name="collision", weight=0.25)
    collision_results = collision_metric.compute(
        log_agents_data.collision,
        agents_data.collision,
        log_agents_data.mask,
    )
    results.update(collision_results)
    # collision_metric.plot_histograms(logs_collision, agents_collision, log_agents_mask)

    # 2.2 TTC
    # TODO: Implement TTC metric

    # 2.3 Object distance
    object_distance_metric = HistogramIntersectionMetric(
        min_val=0.0, max_val=40.0, n_bins=10, name="object_distance", weight=0.15
    )
    object_distance_results = object_distance_metric.compute(
        log_agents_data.object_distance,
        agents_data.object_distance,
        log_agents_data.mask,
    )
    results.update(object_distance_results)

    # 3. Map based
    # 3.1 Offroad
    offroad_metric = BinaryHistogramIntersectionMetric(name="offroad", weight=0.25)
    offroad_results = offroad_metric.compute(
        log_agents_data.offroad,
        agents_data.offroad,
        log_agents_data.mask,
    )
    results.update(offroad_results)

    # 3.2 lane center distance
    center_distance_metric = HistogramIntersectionMetric(
        min_val=0.0, max_val=10.0, n_bins=10, name="center_distance", weight=0.15
    )
    center_distance_results = center_distance_metric.compute(
        log_agents_data.center_distance,
        agents_data.center_distance,
        log_agents_data.mask,
    )
    results.update(center_distance_results)

    # 3.3 Traffic light compliance
    # TODO: Implement traffic light compliance metric

    results["meta_score"] = sum([score for name, score in results.items() if name.endswith("_score")])

    return results


def _extract_sim_agent_data(
    agents_array: npt.NDArray[np.float64],
    agents_mask: npt.NDArray[np.bool],
    rollout: List[BoxDetectionWrapper],
    map_api: AbstractMap,
) -> SimAgentsData:

    assert agents_array.ndim == 3
    assert agents_array.shape[-1] == len(BoundingBoxSE2Index)
    assert agents_array.shape[1] == len(rollout)

    # 1. Kinematics
    speed = _get_linear_speed_from_agents_array(agents_array, agents_mask)
    acceleration = _get_linear_acceleration_from_agents_array(agents_array, agents_mask)
    yaw_rate = _get_yaw_rate_from_agents_array(agents_array, agents_mask)
    yaw_acceleration = _get_yaw_acceleration_from_agents_array(agents_array, agents_mask)

    # 2. Interaction based
    collision = _get_collision_feature(agents_array, rollout)
    object_distance = _get_object_distance_feature(agents_array, agents_mask, rollout)

    # 3. Map based
    offroad = _get_offroad_feature(agents_array, agents_mask, map_api)
    center_distance = _get_road_center_distance_feature(agents_array, agents_mask, map_api)

    return SimAgentsData(
        mask=agents_mask,
        speed=speed,
        acceleration=acceleration,
        yaw_rate=yaw_rate,
        yaw_acceleration=yaw_acceleration,
        collision=collision,
        object_distance=object_distance,
        offroad=offroad,
        center_distance=center_distance,
    )


def _collision_rate(log_agents_data: SimAgentsData, agents_data: SimAgentsData) -> Dict[str, float]:

    def _collision_rate(agents_data: SimAgentsData) -> npt.NDArray[np.bool_]:
        return np.any(agents_data.collision, where=agents_data.mask, axis=1).mean()

    return {
        "log_collision_rate": _collision_rate(log_agents_data),
        "agents_collision_rate": _collision_rate(agents_data),
    }


def _offroad_rate(log_agents_data: SimAgentsData, agents_data: SimAgentsData) -> Dict[str, float]:

    def _offroad_rate(agents_data_: SimAgentsData) -> npt.NDArray[np.bool_]:
        return np.any(agents_data_.offroad, where=agents_data_.mask, axis=1).mean()

    return {
        "log_offroad_rate": _offroad_rate(log_agents_data),
        "agents_offroad_rate": _offroad_rate(agents_data),
    }
