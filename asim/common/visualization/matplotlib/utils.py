from functools import lru_cache
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity as affinity
import shapely.geometry as geom

from asim.common.geometry.base import StateSE2, StateSE3
from asim.common.visualization.color.color import WHITE
from asim.common.visualization.color.config import PlotConfig


def add_shapely_polygon_to_ax(
    ax: plt.Axes,
    polygon: geom.Polygon,
    plot_config: PlotConfig,
    disable_smoothing: bool = False,
) -> plt.Axes:
    """
    Adds shapely polygon to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param polygon: shapely Polygon
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """

    def _add_element_helper(element: geom.Polygon):
        """Helper to add single polygon to ax"""
        if plot_config.smoothing_radius is not None and not disable_smoothing:
            element = element.buffer(-plot_config.smoothing_radius).buffer(plot_config.smoothing_radius)
        exterior_x, exterior_y = element.exterior.xy

        if plot_config.shadow:
            shadow_offset = 0.5
            shadow_x = [x + shadow_offset for x in exterior_x]
            shadow_y = [y - shadow_offset for y in exterior_y]
            ax.fill(
                shadow_x,
                shadow_y,
                color="gray",
                alpha=1.0,
                edgecolor=None,
                linewidth=0,
                zorder=plot_config.zorder,
            )

        ax.fill(
            exterior_x,
            exterior_y,
            color=plot_config.fill_color.hex,
            alpha=plot_config.fill_color_alpha,
            edgecolor=plot_config.line_color.hex,
            linewidth=plot_config.line_width,
            zorder=plot_config.zorder,
        )

        for interior in element.interiors:
            x_interior, y_interior = interior.xy
            ax.fill(
                x_interior,
                y_interior,
                color=WHITE.hex,
                alpha=plot_config.fill_color_alpha,
                edgecolor=plot_config.line_color.hex,
                linewidth=plot_config.line_width,
                zorder=plot_config.zorder,
            )

    if isinstance(polygon, geom.Polygon):
        _add_element_helper(polygon)
    elif isinstance(polygon, geom.MultiPolygon):
        for element in polygon:
            _add_element_helper(element)
    else:
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(polygon)}")

    return ax


def add_shapely_linestring_to_ax(
    ax: plt.Axes,
    linestring: geom.LineString,
    plot_config: PlotConfig,
) -> plt.Axes:
    """
    Adds shapely linestring (polyline) to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param linestring: shapely LineString
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """

    x, y = linestring.xy
    ax.plot(
        x,
        y,
        color=plot_config.line_color.hex,
        alpha=plot_config.line_color_alpha,
        linewidth=plot_config.line_width,
        linestyle=plot_config.line_style,
        zorder=plot_config.zorder,
    )
    return ax


@lru_cache(maxsize=32)
def get_pose_triangle(size: float) -> geom.Polygon:
    """Create a triangle shape for the pose."""
    half_size = size / 2
    return geom.Polygon(
        [
            [-half_size, -half_size],
            [half_size, 0],
            [-half_size, half_size],
            [-size / 4, 0],
        ]
    )


def shapely_geometry_local_coords(
    geometry: geom.base.BaseGeometry, origin: Union[StateSE2, StateSE3]
) -> geom.base.BaseGeometry:
    """Helper for transforming shapely geometry in coord-frame"""
    # TODO: move somewhere else for general use
    cos, sin = np.cos(-origin.yaw), np.sin(-origin.yaw)
    xoff, yoff = origin.x, origin.y
    rotated_geometry = affinity.affine_transform(geometry, [cos, sin, -sin, cos, 0, 0])
    translated_geometry = affinity.affine_transform(rotated_geometry, [1, 0, 0, 1, xoff, yoff])
    return translated_geometry


def add_non_repeating_legend_to_ax(ax: plt.Axes) -> plt.Axes:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    return ax
