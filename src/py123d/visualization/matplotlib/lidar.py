import logging
from typing import Literal, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from py123d.datatypes import Lidar
from py123d.datatypes.sensors.lidar_segmentation_label import DefaultLidarSegmentationLabel, LidarSegmentationLabel
from py123d.visualization.color.default import DEFAULT_LIDAR_SEGMENTATION_COLORS

logger = logging.getLogger(__name__)


def _continuous_colormap(
    values: npt.NDArray,
    cmap_name: str = "viridis",
    vmin: float = None,
    vmax: float = None,
    cmap_range: Tuple[float, float] = (0.0, 1.0),
) -> npt.NDArray[np.uint8]:
    """Map continuous values to RGB colors using a matplotlib colormap.

    By default the 10th/90th percentiles of the values span the full colormap and
    values outside saturate at the palette ends. This spends the whole palette on the
    bulk of the data instead of letting a few outliers compress it.

    :param values: 1D array of continuous values.
    :param cmap_name: Name of the matplotlib colormap to use.
    :param vmin: Minimum value for normalization. Defaults to the 10th percentile.
    :param vmax: Maximum value for normalization. Defaults to the 90th percentile.
    :param cmap_range: Fraction of the colormap to use, as (low, high). Use to cut off
        palette ends that are too dark to see against the viewer background.
    :return: Nx3 array of RGB uint8 values.
    """
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    min_val = vmin if vmin is not None else float(np.quantile(values, 0.10))
    max_val = vmax if vmax is not None else float(np.quantile(values, 0.90))
    if max_val - min_val < 1e-8:
        normalized = np.zeros_like(values, dtype=np.float64)
    else:
        normalized = np.clip((values - min_val) / (max_val - min_val), 0.0, 1.0)
    range_low, range_high = cmap_range
    if (range_low, range_high) != (0.0, 1.0):
        normalized = range_low + normalized * (range_high - range_low)
    colormap = plt.get_cmap(cmap_name)
    colors = colormap(normalized)
    return (colors[:, :3] * 255).astype(np.uint8)


def _discrete_colormap(values: npt.NDArray, cmap_name: str = "tab20") -> npt.NDArray[np.uint8]:
    """Map discrete class values to RGB colors using a qualitative colormap.

    Colors are assigned to the values *present in this frame*, so the same id may render with a
    different color across frames. Use :func:`_discrete_colormap_by_id` when stable per-class colors
    are required (e.g. semantic segmentation).

    :param values: 1D array of discrete class labels (e.g. uint8 IDs).
    :param cmap_name: Name of the qualitative matplotlib colormap to use.
    :return: Nx3 array of RGB uint8 values.
    """
    unique_classes, inverse_indices = np.unique(values, return_inverse=True)
    n_classes = len(unique_classes)
    colormap = plt.get_cmap(cmap_name, n_classes)
    class_colors = colormap(np.linspace(0, 1, n_classes))[:, :3]
    colors = class_colors[inverse_indices]
    return (colors * 255).astype(np.uint8)


def _discrete_colormap_by_id(values: npt.NDArray, cmap_name: str = "tab20") -> npt.NDArray[np.uint8]:
    """Map class ids to *stable* RGB colors: a given id always yields the same color, frame-to-frame.

    Unlike :func:`_discrete_colormap`, the color is a deterministic function of the raw class id
    (``id`` indexes a fixed qualitative palette, cycling if there are more classes than palette
    entries). This keeps e.g. "vehicle" the same color in every frame for segmentation overlays.

    :param values: 1D array of class ids (dataset-native, e.g. a ``LidarSegmentationLabel`` value).
    :param cmap_name: Name of the qualitative matplotlib colormap to use.
    :return: Nx3 array of RGB uint8 values.
    """
    colormap = plt.get_cmap(cmap_name)
    palette = (np.array([colormap(i % colormap.N)[:3] for i in range(colormap.N)]) * 255).astype(np.uint8)
    ids = np.asarray(values).astype(np.int64) % colormap.N
    return palette[ids]


def _get_lidar_segmentation_label_class(lidar: Lidar) -> Optional[Type[LidarSegmentationLabel]]:
    """Return the lidar's per-point segmentation taxonomy, if any sensor in it is annotated."""
    label_class: Optional[Type[LidarSegmentationLabel]] = None
    for metadata in lidar.lidar_metadatas.values():
        if metadata.segmentation_label_class is not None:
            label_class = metadata.segmentation_label_class
            break
    return label_class


def _segmentation_colormap(values: npt.NDArray, label_class: Type[LidarSegmentationLabel]) -> npt.NDArray[np.uint8]:
    """Color raw dataset semantic ids with the Cityscapes palette via their unified default label.

    Each raw class id is mapped to its :class:`DefaultLidarSegmentationLabel` (``to_default()``) and then
    to the canonical Cityscapes-palette color (``DEFAULT_LIDAR_SEGMENTATION_COLORS``). Ids unknown to the
    taxonomy fall back to the ``OTHER`` color.

    :param values: 1D array of raw, dataset-native semantic class ids.
    :param label_class: The dataset's :class:`LidarSegmentationLabel` enum.
    :return: Nx3 array of RGB uint8 values.
    """
    ids = np.asarray(values).astype(np.int64)
    member_ids = [int(member) for member in label_class]
    lut_size = max(int(ids.max()) if ids.size else 0, max(member_ids) if member_ids else 0) + 1

    other_rgb = np.array(DEFAULT_LIDAR_SEGMENTATION_COLORS[DefaultLidarSegmentationLabel.OTHER].rgb, dtype=np.uint8)
    lut = np.tile(other_rgb, (lut_size, 1))
    for member in label_class:
        color = DEFAULT_LIDAR_SEGMENTATION_COLORS.get(member.to_default())
        if color is not None:
            lut[int(member)] = color.rgb

    return lut[np.clip(ids, 0, lut_size - 1)]


def get_lidar_pc_color(
    lidar: Lidar,
    color_feature: Literal[
        "none",
        "height",
        "distance",
        "ids",
        "intensity",
        "channel",
        "timestamps",
        "range",
        "elongation",
        "semantic",
        "instance",
    ] = "none",
    dark_mode: bool = False,
) -> npt.NDArray[np.uint8]:
    """Compute per-point RGB colors for a lidar point cloud based on a feature.

    :param lidar: Lidar object containing the point cloud and its metadata.
    :param color_feature: The feature to color the point cloud by.
    :param dark_mode: If True, use white as the default color; otherwise use black.
    :return: Nx3 array of RGB uint8 values.
    """
    point_cloud_3d = lidar.point_cloud_3d
    n_points = len(point_cloud_3d)

    default_value = 255 if dark_mode else 0
    default_color = np.ones((n_points, 3), dtype=np.uint8) * default_value

    if color_feature == "none":
        return default_color
    elif color_feature == "height":
        # Turbo spans blue -> green -> yellow -> red (viridis has no red); quantile
        # normalization spreads it over the actual per-frame height distribution.
        # The outer 8%/10% of turbo are cut off: those tails darken to navy/dark red,
        # which is nearly invisible against the dark viewer background.
        return _continuous_colormap(-point_cloud_3d[:, 2], cmap_name="turbo", cmap_range=(0.08, 0.90))
    elif color_feature == "distance":
        distances = -np.linalg.norm(point_cloud_3d, axis=-1)
        distances = np.clip(distances, -50.0, 0.0)
        return _continuous_colormap(distances)

    # Features that require point_cloud_features to be present
    discrete_features = {"ids", "channel", "instance"}
    continuous_features = {"intensity", "timestamps", "range", "elongation"}
    feature_accessor = {
        "ids": lidar.ids,
        "intensity": lidar.intensity,
        "channel": lidar.channel,
        "timestamps": lidar.timestamps,
        "range": lidar.range,
        "elongation": lidar.elongation,
        "semantic": lidar.semantic,
        "instance": lidar.instance,
    }

    values = feature_accessor.get(color_feature)
    if values is None:
        logger.warning(f"LiDAR point cloud does not contain {color_feature} feature. Falling back to black.")
        return default_color

    # Semantic ids are colored with the Cityscapes palette via the dataset taxonomy attached to the
    # lidar metadata (raw id -> unified default label -> color). If the taxonomy is unavailable (e.g. a
    # log converted before lidar carried its taxonomy), fall back to stable per-id colors.
    if color_feature == "semantic":
        label_class = _get_lidar_segmentation_label_class(lidar)
        if label_class is not None:
            return _segmentation_colormap(values, label_class)
        return _discrete_colormap_by_id(values)
    elif color_feature in discrete_features:
        return _discrete_colormap(values)
    elif color_feature in continuous_features:
        if values.dtype == np.uint8:
            values = values.astype(np.float32)
        elif values.dtype == np.int64:
            values = values.astype(np.float64)
        return _continuous_colormap(values)

    raise ValueError(f"Unknown feature: {color_feature}")
