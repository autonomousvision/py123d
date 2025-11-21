from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
from matplotlib import patches
from matplotlib.path import Path
from shapely import affinity

from py123d.geometry import PoseSE2, PoseSE3
from py123d.visualization.color.config import PlotConfig


def add_shapely_polygon_to_ax(
    ax: plt.Axes,
    polygon: geom.Polygon,
    plot_config: PlotConfig,
    disable_smoothing: bool = False,
    label: Optional[str] = None,
) -> plt.Axes:
    """Adds shapely polygon to birds-eye-view visualization with proper hole handling

    :param ax: matplotlib ax object
    :param polygon: shapely Polygon
    :param plot_config: dictionary containing plot parameters
    :param disable_smoothing: whether to overwrite smoothing of the polygon
    :return: ax with plot
    """

    def _add_element_helper(element: geom.Polygon):
        """Helper to add single polygon to ax with proper holes"""
        if plot_config.smoothing_radius is not None and not disable_smoothing:
            element = element.buffer(-plot_config.smoothing_radius).buffer(plot_config.smoothing_radius)

        # Create path with exterior and interior rings
        def create_polygon_path(polygon):
            # Get exterior coordinates
            # NOTE: Only take first two dimensions in case of 3D coords
            exterior_coords = np.array(polygon.exterior.coords)[:, :2].tolist()

            # Start with exterior ring
            vertices_2d = exterior_coords
            codes = [Path.MOVETO] + [Path.LINETO] * (len(exterior_coords) - 2) + [Path.CLOSEPOLY]

            # Add interior rings (holes)
            for interior in polygon.interiors:
                interior_coords = list(interior.coords)
                vertices_2d.extend(interior_coords)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(interior_coords) - 2) + [Path.CLOSEPOLY])

            return Path(vertices_2d, codes)

        path = create_polygon_path(element)

        # Add main polygon with holes
        patch = patches.PathPatch(
            path,
            facecolor=plot_config.fill_color.hex,
            alpha=plot_config.fill_color_alpha,
            edgecolor=plot_config.line_color.hex,
            linewidth=plot_config.line_width,
            zorder=plot_config.zorder,
            label=label,
        )
        ax.add_patch(patch)

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
    label: Optional[str] = None,
) -> plt.Axes:
    """Adds shapely linestring (polyline) to birds-eye-view visualization

    :param ax: matplotlib ax object
    :param linestring: shapely LineString
    :param plot_config: dictionary containing plot parameters
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
        label=label,
    )
    return ax


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
    geometry: geom.base.BaseGeometry, origin: Union[PoseSE2, PoseSE3]
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
