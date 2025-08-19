from d123.dataset.scene.scene_builder import ArrowSceneBuilder
from d123.dataset.scene.scene_filter import SceneFilter

from d123.common.multithreading.worker_sequential import Sequential
# from d123.common.multithreading.worker_ray import RayDistributed

import os, psutil

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm

from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.common.geometry.base import Point2D, StateSE2
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE2
from d123.common.visualization.color.default import EGO_VEHICLE_CONFIG
from d123.common.visualization.matplotlib.observation import (
    add_bounding_box_to_ax,
    add_box_detections_to_ax,
    add_default_map_on_ax,
    add_traffic_lights_to_ax,
    add_ego_vehicle_to_ax,
)
from d123.dataset.arrow.conversion import TrafficLightDetectionWrapper
from d123.dataset.maps.abstract_map import AbstractMap
from d123.common.datatypes.detection.detection import BoxDetectionWrapper
from d123.dataset.scene.abstract_scene import AbstractScene
import io
from PIL import Image



def _plot_scene_on_ax(
    ax: plt.Axes,
    map_api: AbstractMap,
    ego_state: EgoStateSE2,
    initial_ego_state: Optional[EgoStateSE2],
    box_detections: BoxDetectionWrapper,
    traffic_light_detections: TrafficLightDetectionWrapper,
    radius: float = 120,
) -> plt.Axes:

    if initial_ego_state is not None:
        point_2d = initial_ego_state.center.point_2d
    else:
        point_2d = ego_state.center.point_2d
    add_default_map_on_ax(ax, map_api, point_2d, radius=radius)
    add_traffic_lights_to_ax(ax, traffic_light_detections, map_api)

    add_box_detections_to_ax(ax, box_detections)
    add_ego_vehicle_to_ax(ax, ego_state)

    ax.set_xlim(point_2d.x - radius, point_2d.x + radius)
    ax.set_ylim(point_2d.y - radius, point_2d.y + radius)

    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_scene_to_image(
    map_api: AbstractMap,
    ego_state: EgoStateSE2,
    initial_ego_state: Optional[EgoStateSE2],
    box_detections: BoxDetectionWrapper,
    traffic_light_detections: TrafficLightDetectionWrapper,
    radius: float = 120,
    figsize: Tuple[int, int] = (8, 8),
) -> Image:

    fig, ax = plt.subplots(figsize=figsize)
    _plot_scene_on_ax(ax, map_api, ego_state, initial_ego_state, box_detections, traffic_light_detections, radius)
    ax.set_aspect("equal", adjustable="box")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")


split = "kitti360_detection_all_and_vel"
scene_tokens = None
log_names = None

scene_filter = SceneFilter(
    split_names=[split], log_names=log_names, scene_tokens=scene_tokens, duration_s=15.1, history_s=1.0
)
scene_builder = ArrowSceneBuilder("/data/jbwang/d123/data2/")
worker = Sequential()
# worker = RayDistributed()
scenes = scene_builder.get_scenes(scene_filter, worker)

print(len(scenes))

for scene in scenes[:10]:
    print(scene.log_name, scene.token)

from d123.dataset.arrow.conversion import DetectionType
from d123.simulation.gym.gym_env import GymEnvironment
from d123.simulation.observation.agents_observation import _filter_agents_by_type

import time

images = []
agent_rollouts = []
plot: bool = True
action = [1.0, -0.0]  # Placeholder action, replace with actual action logic
env = GymEnvironment(scenes)

start = time.time()

map_api, ego_state, detection_observation, current_scene = env.reset(scenes[1460])
initial_ego_state = ego_state
cars, _, _ = _filter_agents_by_type(detection_observation.box_detections, detection_types=[DetectionType.VEHICLE])
agent_rollouts.append(BoxDetectionWrapper(cars))
if plot:
    images.append(
        plot_scene_to_image(
            map_api,
            ego_state,
            initial_ego_state,
            detection_observation.box_detections,
            detection_observation.traffic_light_detections,
        )
    )


for i in range(160):
    ego_state, detection_observation, end = env.step(action)
    cars, _, _ = _filter_agents_by_type(detection_observation.box_detections, detection_types=[DetectionType.VEHICLE])
    agent_rollouts.append(BoxDetectionWrapper(cars))
    if plot:
        images.append(
            plot_scene_to_image(
                map_api,
                ego_state,
                initial_ego_state,
                detection_observation.box_detections,
                detection_observation.traffic_light_detections,
            )
        )
    if end:
        print("End of scene reached.")
        break

time_s = time.time() - start
print(time_s)
print(151/ time_s)

import numpy as np


def create_gif(images, output_path, duration=100):
    """
    Create a GIF from a list of PIL images.

    Args:
        images (list): List of PIL.Image objects.
        output_path (str): Path to save the GIF.
        duration (int): Duration between frames in milliseconds.
    """
    if images:
        print(len(images))
        images_p = [img.convert("P", palette=Image.ADAPTIVE) for img in images]
        images_p[0].save(output_path, save_all=True, append_images=images_p[1:], duration=duration, loop=0)


create_gif(images, f"/data/jbwang/d123/data2/{split}_{current_scene.token}.gif", duration=20)