from typing import List
import matplotlib.pyplot as plt

from asim.common.geometry.base import Point2D
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.map_datatypes import MapObjectType


colors = {
    MapObjectType.LANE: "blue",
    MapObjectType.LANE_GROUP: "green",
    MapObjectType.INTERSECTION: "red",
    MapObjectType.CROSSWALK: "orange",
    MapObjectType.CARPARK: "purple",
    MapObjectType.GENERIC_DRIVABLE: "brown",
}


def _plot_map_on_ax(ax: plt.Axes, map_api: AbstractMap, point_2d: Point2D):

    layers: List[MapObjectType] = [MapObjectType.LANE]

    map_objects_dict = map_api.get_proximal_map_objects(point_2d, radius=1000, layers=layers)
    for layer, map_objects in map_objects_dict.items():
        for map_object in map_objects:
            ax.plot(*map_object.shapely_polygon.exterior.xy, color=colors[layer])
    ax.set_title(f"Map: {map_api.map_name}")

