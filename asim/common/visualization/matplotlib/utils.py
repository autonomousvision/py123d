from functools import lru_cache
from typing import Any, Dict

import matplotlib.pyplot as plt
import shapely.geometry as geom

from asim.common.visualization.color.color import WHITE
from asim.common.visualization.color.config import PlotConfig


def add_shapely_polygon_to_ax(ax: plt.Axes, polygon: geom.Polygon, plot_config: PlotConfig) -> plt.Axes:
    """
    Adds shapely polygon to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param polygon: shapely Polygon
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """

    def _add_element_helper(element: geom.Polygon):
        """Helper to add single polygon to ax"""
        exterior_x, exterior_y = element.exterior.xy
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


def add_linestring_to_ax(
    ax: plt.Axes,
    linestring: geom.LineString,
    config: Dict[str, Any],
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
        y,
        x,
        color=config["line_color"].hex,
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )
    return ax


@lru_cache(maxsize=128)
def get_pose_triangle(size: float) -> geom.Polygon:
    """Create a triangle shape for the pose."""
    half_size = size / 2
    return geom.Polygon(
        [
            [-half_size, -half_size],
            [0, half_size],
            [half_size, -half_size],
            [0, -size / 4],
        ]
    )


# def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
#     """Helper for transforming shapely geometry in coord-frame"""
#     a = np.cos(origin.heading)
#     b = np.sin(origin.heading)
#     d = -np.sin(origin.heading)
#     e = np.cos(origin.heading)
#     xoff = -origin.x
#     yoff = -origin.y
#     translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
#     rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
#     return rotated_geometry
