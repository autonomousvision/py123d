from typing import List, Union

import numpy as np
import shapely
from shapely import LinearRing, LineString, Polygon, box, union_all
from shapely.strtree import STRtree


def get_road_edge_linear_rings(
    drivable_polygons: List[Polygon],
    buffer_distance: float = 0.05,
    add_interiors: bool = True,
) -> List[LinearRing]:
    """
    Helper function to extract road edges (i.e. linear rings) from drivable area polygons.
    TODO: Move and rename for general use.
    """

    def _polygon_to_linear_rings(polygon: Polygon) -> List[LinearRing]:
        assert polygon.geom_type == "Polygon"
        linear_ring_list = []
        linear_ring_list.append(polygon.exterior)
        if add_interiors:
            for interior in polygon.interiors:
                linear_ring_list.append(interior)
        return linear_ring_list

    union_polygon = union_all([polygon.buffer(buffer_distance, join_style=2) for polygon in drivable_polygons]).buffer(
        -buffer_distance, join_style=2
    )

    linear_ring_list = []
    if union_polygon.geom_type == "Polygon":
        for polyline in _polygon_to_linear_rings(union_polygon):
            linear_ring_list.append(LinearRing(polyline))
    elif union_polygon.geom_type == "MultiPolygon":
        for polygon in union_polygon.geoms:
            for polyline in _polygon_to_linear_rings(polygon):
                linear_ring_list.append(LinearRing(polyline))

    return linear_ring_list


def split_line_geometry_by_max_length(
    geometries: Union[LineString, LinearRing, List[Union[LineString, LinearRing]]],
    max_length_meters: float,
) -> List[LineString]:
    """
    Splits LineString or LinearRing geometries into smaller segments based on a maximum length.
    TODO: Move and rename for general use.
    """

    if not isinstance(geometries, list):
        geometries = [geometries]

    all_segments = []
    for geom in geometries:
        if geom.length <= max_length_meters:
            all_segments.append(LineString(geom.coords))
            continue

        num_segments = int(np.ceil(geom.length / max_length_meters))
        segment_length = geom.length / num_segments

        for i in range(num_segments):
            start_dist = i * segment_length
            end_dist = min((i + 1) * segment_length, geom.length)
            segment = shapely.ops.substring(geom, start_dist, end_dist)
            all_segments.append(segment)

    return all_segments


def split_polygon_by_grid(polygon: Polygon, cell_size: float) -> List[Polygon]:
    """
    Split a polygon by grid-like cells of given size.
    TODO: Move and rename for general use.
    """

    minx, miny, maxx, maxy = polygon.bounds

    # Generate all grid cells
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    grid_cells = [box(x, y, x + cell_size, y + cell_size) for x in x_coords for y in y_coords]

    # Build spatial index for fast queries
    tree = STRtree(grid_cells)

    # Query cells that potentially intersect
    candidate_indices = tree.query(polygon, predicate="intersects")

    cells = []
    for idx in candidate_indices:
        cell = grid_cells[idx]
        intersection = polygon.intersection(cell)

        if intersection.is_empty:
            continue

        if intersection.geom_type == "Polygon":
            cells.append(intersection)
        elif intersection.geom_type == "MultiPolygon":
            cells.extend(intersection.geoms)

    return cells
