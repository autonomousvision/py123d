from typing import Final, List, Union

import numpy as np
import shapely
from shapely import LinearRing, LineString, Polygon, union_all

ROAD_EDGE_BUFFER: Final[float] = 0.05


def get_road_edge_linear_rings(drivable_polygons: List[Polygon]) -> List[LinearRing]:

    def _polygon_to_linear_rings(polygon: Polygon) -> List[LinearRing]:
        assert polygon.geom_type == "Polygon"
        linear_ring_list = []
        linear_ring_list.append(polygon.exterior)
        for interior in polygon.interiors:
            linear_ring_list.append(interior)
        return linear_ring_list

    union_polygon = union_all([polygon.buffer(ROAD_EDGE_BUFFER, join_style=2) for polygon in drivable_polygons]).buffer(
        -ROAD_EDGE_BUFFER, join_style=2
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
    geometries: List[Union[LineString, LinearRing]],
    max_length_meters: float,
) -> List[LineString]:
    # TODO: move somewhere more appropriate or implement in Polyline2D, PolylineSE2, etc.

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
