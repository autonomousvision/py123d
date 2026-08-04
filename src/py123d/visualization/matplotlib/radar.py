import logging
from typing import Literal

import numpy as np
import numpy.typing as npt

from py123d.datatypes import Radar
from py123d.datatypes.sensors.radar import RadarFeature
from py123d.visualization.matplotlib.lidar import _continuous_colormap, _discrete_colormap

logger = logging.getLogger(__name__)

RadarColorFeature = Literal[
    "none",
    "height",
    "distance",
    "ids",
    "cluster_id",
    "rcs",
    "velocity",
    "velocity_comp",
    "snr",
    "timestamps",
]


def get_radar_pc_color(
    radar: Radar,
    color_feature: RadarColorFeature = "none",
    dark_mode: bool = False,
) -> npt.NDArray[np.uint8]:
    """Compute per-point RGB colors for a radar point cloud based on a feature.

    Mirrors :func:`py123d.visualization.matplotlib.lidar.get_lidar_pc_color`, but exposes radar-native
    features (RCS, velocity magnitude, SNR, cluster id). Velocity options color by the 2D speed
    magnitude. A feature that is unavailable on the cloud falls back to the default color.

    :param radar: Radar object containing the point cloud and its metadata.
    :param color_feature: The feature to color the point cloud by.
    :param dark_mode: If True, use white as the default color; otherwise use black.
    :return: Nx3 array of RGB uint8 values.
    """
    point_cloud_3d = radar.point_cloud_3d
    n_points = len(point_cloud_3d)

    default_value = 255 if dark_mode else 0
    default_color = np.ones((n_points, 3), dtype=np.uint8) * default_value

    if color_feature == "none":
        return default_color
    elif color_feature == "height":
        # Same palette, quantile normalization, and dark-end cutoff as the lidar height coloring.
        return _continuous_colormap(-point_cloud_3d[:, 2], cmap_name="turbo", cmap_range=(0.08, 0.90))
    elif color_feature == "distance":
        distances = -np.linalg.norm(point_cloud_3d, axis=-1)
        distances = np.clip(distances, -100.0, 0.0)
        return _continuous_colormap(distances)

    # Velocity options color by the 2D (vx, vy) speed magnitude.
    velocity_accessor = {"velocity": radar.velocity, "velocity_comp": radar.velocity_comp}
    if color_feature in velocity_accessor:
        velocity = velocity_accessor[color_feature]
        if velocity is None:
            logger.warning(f"Radar point cloud does not contain {color_feature}. Falling back to default color.")
            return default_color
        return _continuous_colormap(np.linalg.norm(velocity, axis=1).astype(np.float64))

    discrete_features = {"ids", "cluster_id"}
    feature_accessor = {
        "ids": radar.ids,
        "cluster_id": radar.cluster_id,
        "rcs": radar.rcs,
        "timestamps": radar.timestamps,
        "snr": (radar.point_cloud_features or {}).get(RadarFeature.SNR.serialize()),
    }

    values = feature_accessor.get(color_feature)
    if values is None:
        logger.warning(f"Radar point cloud does not contain {color_feature} feature. Falling back to default color.")
        return default_color

    if color_feature in discrete_features:
        return _discrete_colormap(values)

    # Continuous features (rcs, timestamps, snr).
    if values.dtype == np.uint8:
        values = values.astype(np.float32)
    elif values.dtype == np.int64:
        values = values.astype(np.float64)
    return _continuous_colormap(values)
