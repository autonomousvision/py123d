from typing import List

import matplotlib.pyplot as plt

from asim.common.geometry.base import Point2D
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.map_datatypes import MapSurfaceType

colors = {
    MapSurfaceType.LANE: "grey",
    MapSurfaceType.LANE_GROUP: "green",
    MapSurfaceType.INTERSECTION: "red",
    MapSurfaceType.CROSSWALK: "orange",
    MapSurfaceType.CARPARK: "purple",
    MapSurfaceType.GENERIC_DRIVABLE: "brown",
}


def _plot_map_on_ax(ax: plt.Axes, map_api: AbstractMap, point_2d: Point2D, radius: float) -> None:

    layers: List[MapSurfaceType] = [
        MapSurfaceType.LANE,
        MapSurfaceType.GENERIC_DRIVABLE,
        MapSurfaceType.CARPARK,
        MapSurfaceType.CROSSWALK,
    ]
    # layers: List[MapSurfaceType] = [MapSurfaceType.LANE]

    map_objects_dict = map_api.get_proximal_map_objects(point_2d, radius=radius, layers=layers)
    for layer, map_objects in map_objects_dict.items():
        for map_object in map_objects:
            ax.fill(*map_object.shapely_polygon.exterior.xy, color=colors[layer], alpha=0.5)
    ax.set_title(f"Map: {map_api.map_name}")
