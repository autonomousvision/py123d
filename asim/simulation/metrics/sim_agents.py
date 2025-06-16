# NOTE: Do not publish. Waymo licenses suck.


from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from asim.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index
from asim.dataset.arrow.conversion import BoxDetection, DetectionType
from asim.dataset.recording.detection.detection import BoxDetectionWrapper
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.metrics.histogram_metric import HistogramIntersectionMetric


def get_sim_agents_metrics(scene: AbstractScene, agent_rollouts: List[BoxDetectionWrapper]) -> float:
    def get_agent_tokens(agent_rollout: List[BoxDetection]) -> List[str]:
        return [
            box_detection.metadata.track_token
            for box_detection in agent_rollout
            if box_detection.metadata.detection_type == DetectionType.VEHICLE
        ]

    initial_agent_tokens = get_agent_tokens(agent_rollouts[0])

    speed_metric = HistogramIntersectionMetric(min_val=0.0, max_val=25.0, n_bins=10)
    log_agents_array, log_agents_mask = _get_log_agents_array(scene, initial_agent_tokens)
    agents_array, agents_mask = _get_rollout_agents_array(agent_rollouts, initial_agent_tokens)

    # linear_speed_gt = _get_linear_speed_from_agents_array(agents_array)
    # print(linear_speed_gt)

    # return gt_speed, agent_speed
    # return linear_speed_gt
    return agents_array, log_agents_mask


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
