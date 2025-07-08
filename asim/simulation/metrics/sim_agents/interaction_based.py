from typing import Final, List

import numpy as np
import numpy.typing as npt

from asim.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index
from asim.common.geometry.bounding_box.utils import bbse2_array_to_polygon_array
from asim.dataset.arrow.conversion import BoxDetectionWrapper

MAX_OBJECT_DISTANCE: Final[float] = 50.0


def _get_collision_feature(
    agent_array: npt.NDArray[np.float64], box_detections_list: List[BoxDetectionWrapper]
) -> npt.NDArray[np.bool]:
    """
    Extracts the collision feature from the agent array.
    :param agent_array: The agent array containing bounding box information.
    :return: A boolean array indicating collisions.
    """
    assert agent_array.ndim == 3
    assert agent_array.shape[-1] == len(BoundingBoxSE2Index)
    assert agent_array.shape[1] == len(box_detections_list)
    n_objects, n_iterations = agent_array.shape[:2]
    collision_feature = np.zeros((n_objects, n_iterations), dtype=np.bool_)

    agent_polygon_array = bbse2_array_to_polygon_array(agent_array)
    for iteration, box_detections in enumerate(box_detections_list):
        occupancy_map = box_detections.occupancy_map
        for agent_idx in range(n_objects):
            agent_polygon = agent_polygon_array[agent_idx, iteration]
            intersecting_tokens = occupancy_map.intersects(agent_polygon)
            collision_feature[agent_idx, iteration] = len(intersecting_tokens) > 1

    return collision_feature


def _get_object_distance_feature(
    agent_array: npt.NDArray[np.float64],
    agents_mask: npt.NDArray[np.bool],
    box_detections_list: List[BoxDetectionWrapper],
) -> npt.NDArray[np.float64]:
    """
    Extracts the collision feature from the agent array.
    :param agent_array: The agent array containing bounding box information.
    :return: A boolean array indicating collisions.
    """
    assert agent_array.ndim == 3
    assert agent_array.shape[-1] == len(BoundingBoxSE2Index)
    assert agent_array.shape[1] == len(box_detections_list)
    n_objects, n_iterations = agent_array.shape[:2]
    object_distance_feature = np.zeros((n_objects, n_iterations), dtype=np.float64)

    agent_polygon_array = bbse2_array_to_polygon_array(agent_array)
    for iteration, box_detections in enumerate(box_detections_list):
        if agents_mask[:, iteration].any():

            occupancy_map = box_detections.occupancy_map
            _, distances = occupancy_map.query_nearest(
                agent_polygon_array[agents_mask[:, iteration], iteration],
                exclusive=True,
                return_distance=True,
                all_matches=False,
            )
            if len(distances) == 0:
                distances = np.full((n_objects,), MAX_OBJECT_DISTANCE, dtype=np.float64)

            if len(agent_polygon_array[agents_mask[:, iteration]]) == 1:
                distances = min(distances.min(), MAX_OBJECT_DISTANCE)

            object_distance_feature[agents_mask[:, iteration], iteration] = distances

    object_distance_feature = np.clip(object_distance_feature, 0.0, MAX_OBJECT_DISTANCE)

    return object_distance_feature
