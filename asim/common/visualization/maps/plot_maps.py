from typing import List

import matplotlib.pyplot as plt

from asim.common.geometry.base import Point2D
from asim.common.visualization.color.lib.asim import MAP_SURFACE_COLORS
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.abstract_map_objects import AbstractIntersection, AbstractLane
from asim.dataset.maps.map_datatypes import MapSurfaceType


def _plot_map_on_ax(ax: plt.Axes, map_api: AbstractMap, point_2d: Point2D, radius: float) -> None:

    layers: List[MapSurfaceType] = [
        MapSurfaceType.LANE,
        MapSurfaceType.GENERIC_DRIVABLE,
        MapSurfaceType.CARPARK,
        MapSurfaceType.CROSSWALK,
        MapSurfaceType.INTERSECTION,
    ]
    # layers: List[MapSurfaceType] = [MapSurfaceType.LANE]

    map_objects_dict = map_api.get_proximal_map_objects(point_2d, radius=radius, layers=layers)
    for layer, map_objects in map_objects_dict.items():
        for map_object in map_objects:
            ax.fill(*map_object.shapely_polygon.exterior.xy, color=MAP_SURFACE_COLORS[layer].hex, alpha=0.5)
    ax.set_title(f"Map: {map_api.map_name}")


def _plot_map_on_ax_v2(ax: plt.Axes, map_api: AbstractMap, point_2d: Point2D, radius: float) -> None:

    layers: List[MapSurfaceType] = [
        MapSurfaceType.LANE,
        MapSurfaceType.LANE_GROUP,
        MapSurfaceType.GENERIC_DRIVABLE,
        MapSurfaceType.CARPARK,
        MapSurfaceType.CROSSWALK,
        MapSurfaceType.INTERSECTION,
        MapSurfaceType.WALKWAY,
    ]
    # layers: List[MapSurfaceType] = [MapSurfaceType.LANE]

    map_objects_dict = map_api.get_proximal_map_objects(point_2d, radius=radius, layers=layers)
    for layer, map_objects in map_objects_dict.items():
        for map_object in map_objects:
            if layer in [
                MapSurfaceType.GENERIC_DRIVABLE,
                MapSurfaceType.CARPARK,
                MapSurfaceType.CROSSWALK,
                MapSurfaceType.WALKWAY,
            ]:

                ax.fill(
                    *map_object.shapely_polygon.exterior.xy, color=MAP_SURFACE_COLORS[layer].hex, alpha=1.0, zorder=1
                )
            if layer in [MapSurfaceType.LANE_GROUP, MapSurfaceType.INTERSECTION]:
                ax.fill(
                    *map_object.shapely_polygon.exterior.xy, color=MAP_SURFACE_COLORS[layer].hex, alpha=1.0, zorder=0
                )
            if layer in [MapSurfaceType.LANE]:
                map_object: AbstractLane
                ax.plot(*map_object.centerline.array[:, :2].T, color="grey", alpha=1.0, linestyle="dashed", zorder=2)

    ax.set_title(f"Map: {map_api.map_name}")


# def _plot_map_on_ax_v3(ax: plt.Axes, map_api: AbstractMap, point_2d: Point2D, radius: float) -> None:

#     layers: List[MapSurfaceType] = [
#         MapSurfaceType.INTERSECTION,
#     ]
#     # layers: List[MapSurfaceType] = [MapSurfaceType.LANE]

#     map_objects_dict = map_api.get_proximal_map_objects(point_2d, radius=radius, layers=layers)
#     interection: AbstractIntersection = map_objects_dict[MapSurfaceType.INTERSECTION][0]
#     ax.fill(*interection.shapely_polygon.exterior.xy, color="grey", alpha=1.0, label="Intersection")

#     lane_group = interection.lane_groups[0]
#     ax.fill(*lane_group.shapely_polygon.exterior.xy, color="C1", alpha=0.5, label="Lane Group")

#     lane = lane_group.lanes[0]
#     ax.fill(*lane.shapely_polygon.exterior.xy, color="C2", alpha=0.5, label="Lane")

#     ax.plot(*lane.centerline.array[:, :2].T, color="grey", alpha=1.0, linestyle="dashed", zorder=2, label="Lane Centerline")

#     ax.set_title(f"Map: {map_api.map_name}")


def _plot_map_on_ax_v3(ax: plt.Axes, map_api: AbstractMap, point_2d: Point2D, radius: float) -> None:

    layers: List[MapSurfaceType] = [
        MapSurfaceType.INTERSECTION,
    ]
    # layers: List[MapSurfaceType] = [MapSurfaceType.LANE]

    map_objects_dict = map_api.get_proximal_map_objects(point_2d, radius=radius, layers=layers)
    interection: AbstractIntersection = map_objects_dict[MapSurfaceType.INTERSECTION][0]
    # ax.fill(*interection.shapely_polygon.exterior.xy, color="grey", alpha=1.0, label="Intersection")

    interection.lane_groups[0]
    counter = 0
    for idx, element in enumerate(interection.lane_groups):
        for idx, lane in enumerate(element.lanes):
            ax.fill(*lane.shapely_polygon.exterior.xy, alpha=0.5, label=f"Lane {counter}")
            counter += 1

    # lane = lane_group.lanes[0]
    # ax.fill(*lane.shapely_polygon.exterior.xy, color="C2", alpha=0.5, label="Lane")

    # ax.plot(*lane.centerline.array[:, :2].T, color="grey", alpha=1.0, linestyle="dashed", zorder=2, label="Lane Centerline")

    ax.set_title(f"Map: {map_api.map_name}")
