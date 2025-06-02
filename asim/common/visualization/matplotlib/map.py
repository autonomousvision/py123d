from typing import List

import matplotlib.pyplot as plt

from asim.common.geometry.base import Point2D
from asim.common.visualization.color.default import MAP_SURFACE_CONFIG
from asim.common.visualization.matplotlib.utils import add_shapely_polygon_to_ax
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.abstract_map_objects import AbstractLane
from asim.dataset.maps.map_datatypes import MapSurfaceType


def plot_default_map_on_ax(ax: plt.Axes, map_api: AbstractMap, point_2d: Point2D, radius: float) -> None:

    layers: List[MapSurfaceType] = [
        MapSurfaceType.LANE,
        MapSurfaceType.LANE_GROUP,
        MapSurfaceType.GENERIC_DRIVABLE,
        MapSurfaceType.CARPARK,
        MapSurfaceType.CROSSWALK,
        MapSurfaceType.INTERSECTION,
        MapSurfaceType.WALKWAY,
    ]

    map_objects_dict = map_api.get_proximal_map_objects(point_2d, radius=radius, layers=layers)
    for layer, map_objects in map_objects_dict.items():
        for map_object in map_objects:
            if layer in [
                MapSurfaceType.LANE_GROUP,
                MapSurfaceType.GENERIC_DRIVABLE,
                MapSurfaceType.CARPARK,
                MapSurfaceType.CROSSWALK,
                MapSurfaceType.INTERSECTION,
                MapSurfaceType.WALKWAY,
            ]:
                add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, MAP_SURFACE_CONFIG[layer])
            if layer in [MapSurfaceType.LANE]:
                map_object: AbstractLane
                # TODO: refactor
                ax.plot(*map_object.centerline.array[:, :2].T, color="grey", alpha=1.0, linestyle="dashed", zorder=2)

    ax.set_title(f"Map: {map_api.map_name}")
