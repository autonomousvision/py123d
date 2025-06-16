from typing import List

import numpy as np
import numpy.typing as npt

from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE2Index
from asim.dataset.arrow.conversion import BoxDetectionWrapper


def _get_collision_feature(
    agent_array: npt.NDArray[np.float64], box_detections: List[BoxDetectionWrapper]
) -> npt.NDArray[np.float64]:
    """
    Extracts the collision feature from the agent array.
    :param agent_array: The agent array containing bounding box information.
    :return: A boolean array indicating collisions.
    """
    assert agent_array.ndim == 3
    assert agent_array.shape[-1] == len(BoundingBoxSE2Index)
    n_objects, n_iterations = agent_array.shape[:2]
    collision_feature = np.zeros(agent_array.shape[0], dtype=np.bool_)
