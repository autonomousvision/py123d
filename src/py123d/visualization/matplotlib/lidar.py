from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from py123d.conversion.registry.lidar_index_registry import LiDARIndex


def get_lidar_pc_color(
    lidar_pc: npt.NDArray[np.float32],
    lidar_index: LiDARIndex,
    feature: Literal["none", "distance", "intensity"],
) -> npt.NDArray[np.uint8]:
    """
    Compute color map of lidar point cloud according to global configuration
    :param lidar_pc: numpy array of shape (6,n)
    :param as_hex: whether to return hex values, defaults to False
    :return: list of RGB or hex values
    """

    lidar_xyz = lidar_pc[:, lidar_index.XYZ]
    if feature == "none":
        colors_rgb = np.zeros((len(lidar_xyz), 3), dtype=np.uin8)
    else:
        if feature == "distance":
            color_intensities = np.linalg.norm(lidar_xyz, axis=-1)
        elif feature == "intensity":
            assert lidar_index.INTENSITY is not None, "LiDARIndex.INTENSITY is not defined"
            color_intensities = lidar_pc[:, lidar_index.INTENSITY]

        min, max = color_intensities.min(), color_intensities.max()
        norm_intensities = [(value - min) / (max - min) for value in color_intensities]
        colormap = plt.get_cmap("viridis")
        colors_rgb = np.array([colormap(value) for value in norm_intensities])
        colors_rgb = (colors_rgb[:, :3] * 255).astype(np.uint8)

    return colors_rgb
