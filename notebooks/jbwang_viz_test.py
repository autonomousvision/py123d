# from typing import Tuple

# import matplotlib.pyplot as plt

# from nuplan.planning.utils.multithreading.worker_sequential import Sequential

# from d123.dataset.scene.scene_builder import ArrowSceneBuilder
# from d123.dataset.scene.scene_filter import SceneFilter
# from d123.dataset.scene.abstract_scene import AbstractScene

# from typing import Dict
# from d123.common.datatypes.sensor.camera import CameraType
# from d123.common.visualization.matplotlib.camera import add_camera_ax
# from d123.common.visualization.matplotlib.camera import add_box_detections_to_camera_ax

# # split = "nuplan_private_test"
# # log_names = ["2021.09.29.17.35.58_veh-44_00066_00432"]




# # splits = ["carla"]
# splits = ["nuplan_private_test"]
# # splits = ["wopd_train"]
# # log_names = None



# # splits = ["nuplan_private_test"]
# log_names = None

# scene_tokens = None

# scene_filter = SceneFilter(
#     split_names=splits,
#     log_names=log_names,
#     scene_tokens=scene_tokens,
#     duration_s=19,
#     history_s=0.0,
#     timestamp_threshold_s=20,
#     shuffle=False,
#     camera_types=[CameraType.CAM_F0],
# )
# scene_builder = ArrowSceneBuilder("/data/jbwang/d123/data/")
# worker = Sequential()
# # worker = RayDistributed()
# scenes = scene_builder.get_scenes(scene_filter, worker)

# print(f"Found {len(scenes)} scenes")


# from typing import List, Optional, Tuple
# import matplotlib.pyplot as plt
# import numpy as np
# from d123.common.geometry.base import Point2D
# from d123.common.visualization.color.color import BLACK, DARK_GREY, DARKER_GREY, LIGHT_GREY, NEW_TAB_10, TAB_10
# from d123.common.visualization.color.config import PlotConfig
# from d123.common.visualization.color.default import CENTERLINE_CONFIG, MAP_SURFACE_CONFIG, ROUTE_CONFIG
# from d123.common.visualization.matplotlib.observation import (
#     add_box_detections_to_ax,
#     add_default_map_on_ax,
#     add_ego_vehicle_to_ax,
#     add_traffic_lights_to_ax,
# )
# from d123.common.visualization.matplotlib.utils import add_shapely_linestring_to_ax, add_shapely_polygon_to_ax
# from d123.dataset.maps.abstract_map import AbstractMap
# from d123.dataset.maps.abstract_map_objects import AbstractLane
# from d123.dataset.maps.map_datatypes import MapLayer
# from d123.dataset.scene.abstract_scene import AbstractScene


# import shapely.geometry as geom

# LEFT_CONFIG: PlotConfig = PlotConfig(
#     fill_color=TAB_10[2],
#     fill_color_alpha=1.0,
#     line_color=TAB_10[2],
#     line_color_alpha=0.5,
#     line_width=1.0,
#     line_style="-",
#     zorder=3,
# )

# RIGHT_CONFIG: PlotConfig = PlotConfig(
#     fill_color=TAB_10[3],
#     fill_color_alpha=1.0,
#     line_color=TAB_10[3],
#     line_color_alpha=0.5,
#     line_width=1.0,
#     line_style="-",
#     zorder=3,
# )


# LANE_CONFIG: PlotConfig = PlotConfig(
#     fill_color=BLACK,
#     fill_color_alpha=1.0,
#     line_color=BLACK,
#     line_color_alpha=0.0,
#     line_width=0.0,
#     line_style="-",
#     zorder=5,
# )

# ROAD_EDGE_CONFIG: PlotConfig = PlotConfig(
#     fill_color=DARKER_GREY.set_brightness(0.0),
#     fill_color_alpha=1.0,
#     line_color=DARKER_GREY.set_brightness(0.0),
#     line_color_alpha=1.0,
#     line_width=1.0,
#     line_style="-",
#     zorder=3,
# )

# ROAD_LINE_CONFIG: PlotConfig = PlotConfig(
#     fill_color=DARKER_GREY,
#     fill_color_alpha=1.0,
#     line_color=NEW_TAB_10[5],
#     line_color_alpha=1.0,
#     line_width=1.5,
#     line_style="-",
#     zorder=3,
# )


# def add_debug_map_on_ax(
#     ax: plt.Axes,
#     map_api: AbstractMap,
#     point_2d: Point2D,
#     radius: float,
#     route_lane_group_ids: Optional[List[int]] = None,
# ) -> None:
#     layers: List[MapLayer] = [
#         MapLayer.LANE,
#         MapLayer.LANE_GROUP,
#         MapLayer.GENERIC_DRIVABLE,
#         MapLayer.CARPARK,
#         MapLayer.CROSSWALK,
#         MapLayer.INTERSECTION,
#         MapLayer.WALKWAY,
#         MapLayer.ROAD_EDGE,
#         MapLayer.ROAD_LINE,
#     ]
#     x_min, x_max = point_2d.x - radius, point_2d.x + radius
#     y_min, y_max = point_2d.y - radius, point_2d.y + radius
#     patch = geom.box(x_min, y_min, x_max, y_max)
#     map_objects_dict = map_api.query(geometry=patch, layers=layers, predicate="intersects")

#     done = False
#     for layer, map_objects in map_objects_dict.items():
#         for map_object in map_objects:
#             try:
#                 if layer in [
#                     # MapLayer.GENERIC_DRIVABLE,
#                     # MapLayer.CARPARK,
#                     # MapLayer.CROSSWALK,
#                     # MapLayer.INTERSECTION,
#                     # MapLayer.WALKWAY,
#                 ]:
#                     add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, MAP_SURFACE_CONFIG[layer])

#                 # if layer in [MapLayer.LANE_GROUP]:
#                 #     add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, MAP_SURFACE_CONFIG[layer])

#                 if layer in [MapLayer.LANE]:
#                     map_object: AbstractLane
#                     if map_object.right_lane is not None and map_object.left_lane is not None and not done:
#                         add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, LANE_CONFIG)
#                         add_shapely_polygon_to_ax(ax, map_object.right_lane.shapely_polygon, RIGHT_CONFIG)
#                         add_shapely_polygon_to_ax(ax, map_object.left_lane.shapely_polygon, LEFT_CONFIG)
#                         done = True
#                     else:
#                         add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, MAP_SURFACE_CONFIG[layer])


#                 # add_shapely_linestring_to_ax(ax, map_object.right_boundary.linestring, RIGHT_CONFIG)
#                 # add_shapely_linestring_to_ax(ax, map_object.left_boundary.linestring, LEFT_CONFIG)
#                 # add_shapely_polygon_to_ax(ax, map_object.shapely_polygon, LANE_CONFIG)

#                 # centroid = map_object.shapely_polygon.centroid
#                 # ax.text(
#                 #     centroid.x,
#                 #     centroid.y,
#                 #     str(map_object.id),
#                 #     horizontalalignment="center",
#                 #     verticalalignment="center",
#                 #     fontsize=8,
#                 #     bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"),
#                 # )
#                 # if layer in [MapLayer.ROAD_EDGE]:
#                 #     add_shapely_linestring_to_ax(ax, map_object.polyline_3d.linestring, ROAD_EDGE_CONFIG)
#                 #     edge_lengths.append(map_object.polyline_3d.linestring.length)

#                 if layer in [MapLayer.ROAD_LINE]:
#                     line_type = int(map_object.road_line_type)
#                     plt_config = PlotConfig(
#                         fill_color=NEW_TAB_10[line_type % len(NEW_TAB_10)],
#                         fill_color_alpha=1.0,
#                         line_color=NEW_TAB_10[line_type % len(NEW_TAB_10)],
#                         line_color_alpha=1.0,
#                         line_width=1.5,
#                         line_style="-",
#                         zorder=3,
#                     )
#                     add_shapely_linestring_to_ax(ax, map_object.polyline_3d.linestring, plt_config)

#             except Exception:
#                 import traceback

#                 print(f"Error adding map object of type {layer.name} and id {map_object.id}")
#                 traceback.print_exc()

#     ax.set_title(f"Map: {map_api.map_name}")


# def _plot_scene_on_ax(ax: plt.Axes, scene: AbstractScene, iteration: int = 0, radius: float = 80) -> plt.Axes:

#     ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
#     box_detections = scene.get_box_detections_at_iteration(iteration)

#     point_2d = ego_vehicle_state.bounding_box.center.state_se2.point_2d
#     add_debug_map_on_ax(ax, scene.map_api, point_2d, radius=radius, route_lane_group_ids=None)
#     # add_default_map_on_ax(ax, scene.map_api, point_2d, radius=radius, route_lane_group_ids=None)
#     # add_traffic_lights_to_ax(ax, traffic_light_detections, scene.map_api)

#     add_box_detections_to_ax(ax, box_detections)
#     add_ego_vehicle_to_ax(ax, ego_vehicle_state)

#     zoom = 1.0
#     ax.set_xlim(point_2d.x - radius * zoom, point_2d.x + radius * zoom)
#     ax.set_ylim(point_2d.y - radius * zoom, point_2d.y + radius * zoom)

#     ax.set_aspect("equal", adjustable="box")
#     return ax


# def plot_scene_at_iteration(
#     scene: AbstractScene, iteration: int = 0, radius: float = 80
# ) -> Tuple[plt.Figure, plt.Axes]:

#     size = 15

#     fig, ax = plt.subplots(figsize=(size, size))
#     _plot_scene_on_ax(ax, scene, iteration, radius)
#     return fig, ax


# scene_index = 1
# fig, ax = plot_scene_at_iteration(scenes[scene_index], iteration=100, radius=100)

# # fig.savefig(f"/home/daniel/scene_{scene_index}_iteration_1.pdf", dpi=300, bbox_inches="tight")

