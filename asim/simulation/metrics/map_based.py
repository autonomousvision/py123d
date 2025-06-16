import numpy as np
import numpy.typing as npt
import shapely

from asim.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index
from asim.common.geometry.bounding_box.utils import Corners2DIndex, bbse2_array_to_corners_array
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.map_datatypes import MapSurfaceType


def _get_offroad_feature(
    agents_array: npt.NDArray[np.float64], agents_mask: npt.NDArray[np.bool], map_api: AbstractMap
) -> npt.NDArray[np.float64]:

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
