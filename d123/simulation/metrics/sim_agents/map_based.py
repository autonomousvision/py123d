from typing import Dict, Final, List

import numpy as np
import numpy.typing as npt
import shapely

from d123.common.geometry.base import StateSE2, StateSE2Index
from d123.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index
from d123.common.geometry.bounding_box.utils import Corners2DIndex, bbse2_array_to_corners_array
from d123.common.geometry.utils import normalize_angle
from d123.dataset.maps.abstract_map import AbstractMap
from d123.dataset.maps.abstract_map_objects import AbstractLane
from d123.dataset.maps.map_datatypes import MapSurfaceType

MAX_LANE_CENTER_DISTANCE: Final[float] = 10.0


def _get_offroad_feature(
    agents_array: npt.NDArray[np.float64], agents_mask: npt.NDArray[np.bool], map_api: AbstractMap
) -> npt.NDArray[np.bool]:

    assert agents_array.shape[-1] == len(BoundingBoxSE2Index)
    n_objects, n_iterations = agents_array.shape[:2]

    offroad_feature = np.zeros((n_objects, n_iterations), dtype=np.bool_)

    agent_shapely_corners = shapely.creation.points(bbse2_array_to_corners_array(agents_array)).flatten()
    corner_indices = np.arange(n_iterations * n_objects * len(Corners2DIndex)).reshape(
        n_objects, n_iterations, len(Corners2DIndex)
    )

    output = map_api.query_object_ids(
        agent_shapely_corners,
        layers=[
            MapSurfaceType.INTERSECTION,
            MapSurfaceType.LANE_GROUP,
            MapSurfaceType.CARPARK,
            MapSurfaceType.GENERIC_DRIVABLE,
        ],
        predicate="within",
    )
    list_all_corners = []
    for _, object_ids in output.items():
        list_all_corners.extend(list(object_ids))
    set_of_all_corners = set(list_all_corners)

    for object_idx in range(n_objects):
        for iteration in range(n_iterations):
            if agents_mask[object_idx, iteration]:
                corner_indices_ = set(corner_indices[object_idx, iteration])
                offroad_feature[object_idx, iteration] = not corner_indices_.issubset(set_of_all_corners)

    return offroad_feature


def _get_road_center_distance_feature(
    agents_array: npt.NDArray[np.float64], agents_mask: npt.NDArray[np.bool], map_api: AbstractMap
) -> npt.NDArray[np.float64]:

    lane_dict: Dict[str, AbstractLane] = {}

    def get_lane_by_id(lane_id: str, lane_dict: Dict[str, AbstractLane]) -> AbstractLane:
        if lane_id not in lane_dict.keys():
            lane_dict[lane_id] = map_api.get_map_object(lane_id, MapSurfaceType.LANE)
        return lane_dict[lane_id]

    assert agents_array.shape[-1] == len(BoundingBoxSE2Index)
    n_objects, n_iterations = agents_array.shape[:2]

    agent_shapely_centers = shapely.creation.points(agents_array[..., BoundingBoxSE2Index.XY]).flatten()
    agent_indices = np.arange(n_iterations * n_objects).reshape(n_objects, n_iterations)

    distances = np.full((n_objects, n_iterations), MAX_LANE_CENTER_DISTANCE, dtype=np.float64)

    nearest_query_output = map_api.query_nearest(
        agent_shapely_centers,
        layers=[MapSurfaceType.LANE],
        max_distance=MAX_LANE_CENTER_DISTANCE,
        return_all=True,
        return_distance=False,
        exclusive=False,
    )[MapSurfaceType.LANE]

    for object_idx in range(n_objects):
        for iteration in range(n_iterations):
            agent_idx = agent_indices[object_idx, iteration]

            if (not agents_mask[object_idx, iteration]) or (agent_idx not in nearest_query_output.keys()):
                continue

            lane_ids: List[str] = nearest_query_output[agent_idx]
            lanes: List[AbstractLane] = [get_lane_by_id(lane_id, lane_dict) for lane_id in lane_ids]

            if len(lanes) == 1:
                select_lane = lanes[0]
                centerline = select_lane.centerline.polyline_se2
                projected_se2_array = centerline.interpolate(centerline.project(agent_shapely_centers[agent_idx])).array

            elif len(lanes) > 1:

                projected_se2s_array = np.zeros((len(lanes), len(StateSE2Index)), dtype=np.float64)
                for lane_idx, lane in enumerate(lanes):
                    lane: AbstractLane
                    centerline = lane.centerline.polyline_se2
                    projected_se2s_array[lane_idx] = centerline.interpolate(
                        centerline.project(agent_shapely_centers[agent_idx])
                    ).array
                se2_distances = circumference_distance_se2_array(
                    agents_array[object_idx, iteration, BoundingBoxSE2Index.SE2],
                    projected_se2s_array,
                    radius=agents_array[object_idx, iteration, BoundingBoxSE2Index.LENGTH] / 2,
                )
                projected_se2_array = projected_se2s_array[np.argmin(se2_distances)]
            else:
                raise ValueError

            distances[object_idx, iteration] = np.linalg.norm(
                agents_array[object_idx, iteration, BoundingBoxSE2Index.XY]
                - projected_se2_array[BoundingBoxSE2Index.XY]
            )

    del lane_dict
    return distances


def circumference_distance_se2(state1: StateSE2, state2: StateSE2, radius: float) -> float:
    # TODO: Move this to a more appropriate location for general usage.
    # Heuristic for defining distance/similarity between two SE2 states.
    # Combines the 2D Euclidean distance with the circumference of the yaw difference.
    positional_distance = np.linalg.norm(state1.point_2d.array - state2.point_2d.array)
    abs_yaw_difference = np.abs(normalize_angle(state1.yaw - state2.yaw))
    rotation_distance = abs_yaw_difference * radius
    return positional_distance + rotation_distance


def circumference_distance_se2_array(
    state1_se2: npt.NDArray[np.float64],
    state2_se2: npt.NDArray[np.float64],
    radius: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # TODO: Move this to a more appropriate location for general usage.
    # Heuristic for defining distance/similarity between two SE2 states.
    # Combines the 2D Euclidean distance with the circumference of the yaw difference.
    positional_distance = np.linalg.norm(state1_se2[..., StateSE2Index.XY] - state2_se2[..., StateSE2Index.XY], axis=-1)
    abs_yaw_difference = np.abs(
        normalize_angle(
            state1_se2[..., StateSE2Index.YAW] - state2_se2[..., StateSE2Index.YAW],
        )
    )
    rotation_distance = abs_yaw_difference * radius
    return positional_distance + rotation_distance
