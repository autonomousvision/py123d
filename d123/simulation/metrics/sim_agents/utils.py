from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from d123.common.datatypes.detection.detection import BoxDetectionWrapper
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.geometry.geometry_index import BoundingBoxSE2Index


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
