from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom

from d123.common.datatypes.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2, EgoStateSE3
from d123.common.geometry.base import Point2D
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE2, BoundingBoxSE3
from d123.common.geometry.transform.tranform_2d import translate_along_yaw
from d123.common.visualization.color.config import PlotConfig
from d123.common.visualization.color.default import (
    BOX_DETECTION_CONFIG,
    CENTERLINE_CONFIG,
    EGO_VEHICLE_CONFIG,
    MAP_SURFACE_CONFIG,
    ROUTE_CONFIG,
    TRAFFIC_LIGHT_CONFIG,
)
from d123.common.visualization.matplotlib.utils import (
    add_shapely_linestring_to_ax,
    add_shapely_polygon_to_ax,
    get_pose_triangle,
    shapely_geometry_local_coords,
)
from d123.dataset.maps.abstract_map import AbstractMap
from d123.dataset.maps.abstract_map_objects import AbstractLane
from d123.dataset.maps.map_datatypes import MapLayer


def add_default_map_on_ax(
    ax: plt.Axes,
    map_api: AbstractMap,
    point_2d: Point2D,
    radius: float,
    route_lane_group_ids: Optional[List[int]] = None,
) -> None:
    layers: List[MapLayer] = [
        MapLayer.LANE,
        MapLayer.LANE_GROUP,
        MapLayer.GENERIC_DRIVABLE,
        MapLayer.CARPARK,
        MapLayer.CROSSWALK,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
    ]
    x_min, x_max = point_2d.x - radius, point_2d.x + radius
    y_min, y_max = point_2d.y - radius, point_2d.y + radius
    patch = geom.box(x_min, y_min, x_max, y_max)
    map_objects_dict = map_api.query(geometry=patch, layers=layers, predicate="intersects")

    for layer, map_objects in map_objects_dict.items():
        for map_object in map_objects:
            try:
                if layer in [MapLayer.LANE_GROUP]:
                    if route_lane_group_ids is not None and int(map_object.id) in route_lane_group_ids:
                        add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, ROUTE_CONFIG)
                    else:
                        add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, MAP_SURFACE_CONFIG[layer])
                if layer in [
                    MapLayer.GENERIC_DRIVABLE,
                    MapLayer.CARPARK,
                    MapLayer.CROSSWALK,
                    MapLayer.INTERSECTION,
                    MapLayer.WALKWAY,
                ]:
                    add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, MAP_SURFACE_CONFIG[layer])
                if layer in [MapLayer.LANE]:
                    map_object: AbstractLane
                    add_shapely_linestring_to_ax(ax, map_object.centerline.linestring, CENTERLINE_CONFIG)
            except Exception:
                import traceback

                print(f"Error adding map object of type {layer.name} and id {map_object.id}")
                traceback.print_exc()

    ax.set_title(f"Map: {map_api.map_name}")


def add_box_detections_to_ax(ax: plt.Axes, box_detections: BoxDetectionWrapper) -> None:
    for box_detection in box_detections:
        # TODO: Optionally, continue on boxes outside of plot.
        # if box_detection.metadata.detection_type == DetectionType.GENERIC_OBJECT:
        #     continue
        plot_config = BOX_DETECTION_CONFIG[box_detection.metadata.detection_type]
        add_bounding_box_to_ax(ax, box_detection.bounding_box, plot_config)


def add_ego_vehicle_to_ax(ax: plt.Axes, ego_vehicle_state: Union[EgoStateSE3, EgoStateSE2]) -> None:
    add_bounding_box_to_ax(ax, ego_vehicle_state.bounding_box, EGO_VEHICLE_CONFIG)


def add_traffic_lights_to_ax(
    ax: plt.Axes, traffic_light_detections: TrafficLightDetectionWrapper, map_api: AbstractMap
) -> None:
    for traffic_light_detection in traffic_light_detections:
        lane: AbstractLane = map_api.get_map_object(str(traffic_light_detection.lane_id), MapLayer.LANE)
        if lane is not None:
            add_shapely_linestring_to_ax(
                ax,
                lane.centerline.linestring,
                TRAFFIC_LIGHT_CONFIG[traffic_light_detection.status],
            )
        else:
            raise ValueError(f"Lane with id {traffic_light_detection.lane_id} not found in map {map_api.map_name}.")


def add_bounding_box_to_ax(
    ax: plt.Axes,
    bounding_box: Union[BoundingBoxSE2, BoundingBoxSE3],
    plot_config: PlotConfig,
) -> None:

    add_shapely_polygon_to_ax(ax, bounding_box.shapely_polygon, plot_config)

    if plot_config.marker_style is not None:
        assert plot_config.marker_style in ["-", "^"], f"Unknown marker style: {plot_config.marker_style}"
        if plot_config.marker_style == "-":
            center_se2 = (
                bounding_box.center if isinstance(bounding_box, BoundingBoxSE2) else bounding_box.center.state_se2
            )
            arrow = np.zeros((2, 2), dtype=np.float64)
            arrow[0] = center_se2.point_2d.array
            arrow[1] = translate_along_yaw(center_se2, Point2D(bounding_box.length / 2.0 + 0.5, 0.0)).point_2d.array
            ax.plot(
                arrow[:, 0],
                arrow[:, 1],
                color=plot_config.line_color.hex,
                alpha=plot_config.line_color_alpha,
                linewidth=plot_config.line_width,
                zorder=plot_config.zorder,
                linestyle=plot_config.line_style,
            )
        elif plot_config.marker_style == "^":
            marker_size = min(plot_config.marker_size, min(bounding_box.length, bounding_box.width))
            marker_polygon = get_pose_triangle(marker_size)
            global_marker_polygon = shapely_geometry_local_coords(marker_polygon, bounding_box.center)
            add_shapely_polygon_to_ax(ax, global_marker_polygon, plot_config, disable_smoothing=True)
        else:
            raise ValueError(f"Unknown marker style: {plot_config.marker_style}")
