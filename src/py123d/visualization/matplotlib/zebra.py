# FIXME: currently not used

# import numpy as np
# from shapely.geometry import Polygon

# from py123d.datatypes.map_objects.map_layer_types import MapLayer
# from py123d.visualization.color.color import Color
# from py123d.visualization.color.config import PlotConfig
# from py123d.visualization.color.default import MAP_SURFACE_CONFIG
# from py123d.visualization.matplotlib.utils import add_shapely_polygon_to_ax, add_shapely_polygons_to_ax

# STRIPE_PLOT_CONFIG = PlotConfig(
#     fill_color=Color("#af8da6"),
#     fill_color_alpha=1.0,
#     line_color=Color("#af8da6"),
#     line_color_alpha=0.0,
#     line_width=1.0,
#     line_style="-",
#     zorder=2,
# )


# def visualize_crosswalk_stripes(polygon, ax, num_stripes=8, stripe_width_ratio=0.4):
#     # First, fill the background (dark/gray base)
#     x, y = polygon.exterior.xy
#     ax.fill(x, y, color="#333333", edgecolor="black", linewidth=1.5)

#     # Get the minimum rotated rectangle (oriented bounding box)
#     min_rect = polygon.minimum_rotated_rectangle

#     # Get coordinates of the minimum rectangle
#     rect_coords = np.array(min_rect.exterior.coords[:-1])

#     # Calculate edge lengths to find longitudinal vs lateral direction
#     edge_lengths = [np.linalg.norm(rect_coords[i] - rect_coords[(i + 1) % 4]) for i in range(4)]

#     # Find the longer edge (longitudinal direction)
#     long_edge_idx = 0 if edge_lengths[0] > edge_lengths[1] else 1

#     # Get the direction vector of the longitudinal edge
#     p1 = rect_coords[long_edge_idx]
#     p2 = rect_coords[(long_edge_idx + 1) % 4]
#     longitudinal_vec = p2 - p1
#     longitudinal_length = np.linalg.norm(longitudinal_vec)
#     longitudinal_unit = longitudinal_vec / longitudinal_length

#     # Get the perpendicular (lateral) direction - this is the SHORT side
#     # Stripes should span across this direction
#     p3 = rect_coords[(long_edge_idx + 2) % 4]
#     lateral_vec = p3 - p2
#     lateral_length = np.linalg.norm(lateral_vec)

#     # Calculate stripe dimensions along the longitudinal direction
#     num_stripes = int(longitudinal_length // (0.6 + 1))
#     total_period = longitudinal_length / num_stripes
#     stripe_width = total_period * 0.5

#     # Create stripes perpendicular to longitudinal direction
#     # Each stripe spans the full width (lateral direction)
#     polygons = []
#     for i in range(num_stripes):
#         # Position along the longitudinal direction
#         offset = i * total_period

#         # Create stripe as a thin rectangle perpendicular to longitudinal
#         stripe_start = p1 + longitudinal_unit * offset
#         stripe_end = stripe_start + longitudinal_unit * stripe_width

#         # Create the stripe polygon spanning the lateral direction
#         stripe_poly = Polygon([stripe_start, stripe_end, stripe_end + lateral_vec, stripe_start + lateral_vec])

#         # Intersect with original polygon to respect boundaries
#         stripe_clipped = stripe_poly.intersection(polygon)
#         polygons.append(stripe_clipped)

#     # Plot stripes
#     add_shapely_polygon_to_ax(ax, polygon, plot_config=MAP_SURFACE_CONFIG[MapLayer.CROSSWALK], label="Crosswalk")
#     add_shapely_polygons_to_ax(ax, polygons, plot_config=STRIPE_PLOT_CONFIG)
