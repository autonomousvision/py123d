from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom

from py123d.datatypes.detections.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from py123d.datatypes.detections.detection_types import DetectionType
from py123d.datatypes.maps.abstract_map import AbstractMap
from py123d.datatypes.maps.abstract_map_objects import AbstractLane
from py123d.datatypes.maps.map_datatypes import MapLayer
from py123d.datatypes.scene.abstract_scene import AbstractScene
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE2, EgoStateSE3
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, Point2D, StateSE2Index, Vector2D
from py123d.geometry.transform.transform_se2 import translate_se2_along_body_frame
from py123d.visualization.color.config import PlotConfig
from py123d.visualization.color.default import (
    BOX_DETECTION_CONFIG,
    CENTERLINE_CONFIG,
    EGO_VEHICLE_CONFIG,
    MAP_SURFACE_CONFIG,
    ROUTE_CONFIG,
    TRAFFIC_LIGHT_CONFIG,
)
from py123d.visualization.matplotlib.utils import (
    add_shapely_linestring_to_ax,
    add_shapely_polygon_to_ax,
    get_pose_triangle,
    shapely_geometry_local_coords,
)


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
                    if route_lane_group_ids is not None and int(map_object.object_id) in route_lane_group_ids:
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

                print(f"Error adding map object of type {layer.name} and id {map_object.object_id}")
                traceback.print_exc()

    ax.set_title(f"Map: {map_api.location}")


def add_box_detections_to_ax(ax: plt.Axes, box_detections: BoxDetectionWrapper) -> None:
    for box_detection in box_detections:
        # TODO: Optionally, continue on boxes outside of plot.
        # if box_detection.metadata.detection_type == DetectionType.GENERIC_OBJECT:
        #     continue
        plot_config = BOX_DETECTION_CONFIG[box_detection.metadata.detection_type]
        add_bounding_box_to_ax(ax, box_detection.bounding_box, plot_config)


def add_box_future_detections_to_ax(ax: plt.Axes, scene: AbstractScene, iteration: int) -> None:

    # TODO: Refactor this function
    initial_agents = scene.get_box_detections_at_iteration(iteration)
    agents_poses = {
        agent.metadata.track_token: [agent.center_se3]
        for agent in initial_agents
        if agent.metadata.detection_type == DetectionType.VEHICLE
    }
    frequency = 1
    for iteration in range(iteration + frequency, scene.number_of_iterations, frequency):
        agents = scene.get_box_detections_at_iteration(iteration)
        for agent in agents:
            if agent.metadata.track_token in agents_poses:
                agents_poses[agent.metadata.track_token].append(agent.center_se3)

    for track_token, poses in agents_poses.items():
        if len(poses) < 2:
            continue
        poses = np.array([pose.point_2d.array for pose in poses])
        num_poses = poses.shape[0]
        alphas = 1 - np.linspace(0.2, 1.0, num_poses)  # Start low, end high
        for i in range(num_poses - 1):
            ax.plot(
                poses[i : i + 2, 0],
                poses[i : i + 2, 1],
                color=BOX_DETECTION_CONFIG[DetectionType.VEHICLE].fill_color.hex,
                alpha=alphas[i + 1],
                linewidth=BOX_DETECTION_CONFIG[DetectionType.VEHICLE].line_width * 5,
                zorder=BOX_DETECTION_CONFIG[DetectionType.VEHICLE].zorder,
            )


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
            raise ValueError(f"Lane with id {traffic_light_detection.lane_id} not found in map {map_api.location}.")


def add_bounding_box_to_ax(
    ax: plt.Axes,
    bounding_box: Union[BoundingBoxSE2, BoundingBoxSE3],
    plot_config: PlotConfig,
) -> None:

    add_shapely_polygon_to_ax(ax, bounding_box.shapely_polygon, plot_config)

    if plot_config.marker_style is not None:
        assert plot_config.marker_style in ["-", "^"], f"Unknown marker style: {plot_config.marker_style}"
        if plot_config.marker_style == "-":
            center_se2 = bounding_box.center_se2
            arrow = np.zeros((2, 2), dtype=np.float64)
            arrow[0] = center_se2.point_2d.array
            arrow[1] = translate_se2_along_body_frame(
                center_se2,
                Vector2D(bounding_box.length / 2.0 + 0.5, 0.0),
            ).array[StateSE2Index.XY]
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
