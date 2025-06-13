from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm

from asim.common.visualization.matplotlib.observation import (
    add_box_detections_to_ax,
    add_default_map_on_ax,
    add_ego_vehicle_to_ax,
    add_traffic_lights_to_ax,
)
from asim.dataset.scene.abstract_scene import AbstractScene


def _plot_scene_on_ax(ax: plt.Axes, scene: AbstractScene, iteration: int = 0, radius: float = 80) -> plt.Axes:

    ego_vehicle_state = scene.get_ego_vehicle_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)
    traffic_light_detections = scene.get_traffic_light_detections_at_iteration(iteration)
    route_lane_group_ids = scene.get_route_lane_group_ids(iteration)
    map_api = scene.map_api

    point_2d = ego_vehicle_state.bounding_box.center.state_se2.point_2d
    add_default_map_on_ax(ax, map_api, point_2d, radius=radius, route_lane_group_ids=route_lane_group_ids)
    add_traffic_lights_to_ax(ax, traffic_light_detections, map_api)

    add_box_detections_to_ax(ax, box_detections)
    add_ego_vehicle_to_ax(ax, ego_vehicle_state)

    ax.set_xlim(point_2d.x - radius, point_2d.x + radius)
    ax.set_ylim(point_2d.y - radius, point_2d.y + radius)

    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_scene_at_iteration(
    scene: AbstractScene, iteration: int = 0, radius: float = 80
) -> Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_scene_on_ax(ax, scene, iteration, radius)
    return fig, ax


def render_scene_animation(
    scene: AbstractScene,
    output_path: Union[str, Path],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    step: int = 10,
    fps: float = 20.0,
    dpi: int = 300,
    format: str = "mp4",
    radius: float = 80,
) -> None:
    assert format in ["mp4", "gif"], "Format must be either 'mp4' or 'gif'."
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scene.open()

    if end_idx is None:
        end_idx = scene.get_number_of_iterations()
    end_idx = min(end_idx, scene.get_number_of_iterations())

    fig, ax = plt.subplots(figsize=(10, 10))

    def update(i):
        ax.clear()
        _plot_scene_on_ax(ax, scene, i, radius)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        pbar.update(1)

    frames = list(range(start_idx, end_idx, step))
    pbar = tqdm(total=len(frames), desc=f"Rendering {scene.log_name} as MP4")
    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)

    ani.save(output_path / f"{scene.log_name}_{scene.token}.{format}", writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    scene.close()
