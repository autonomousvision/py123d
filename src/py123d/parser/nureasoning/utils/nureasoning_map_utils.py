"""Geometry helpers for the nuReasoning map parser.

nuReasoning lanes ship a closed polygon outline plus a centerline, but py123d lanes require explicit
left and right boundaries. ``split_polygon_into_boundaries`` derives those two boundaries by assigning
each polygon vertex to the left or right side of the centerline.

The split / ordering helpers below are adapted from ``parser/nuscenes/utils/nuscenes_map_utils.py``
(originally from trajdata, Apache 2.0 License). They are duplicated here rather than imported because the
nuScenes module hard-requires the nuScenes devkit at import time, which the nuReasoning parser does not
depend on.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from py123d.geometry.polyline import Polyline2D


def split_polygon_into_boundaries(
    polygon_xy: np.ndarray, centerline_xy: np.ndarray
) -> Tuple[Optional[Polyline2D], Optional[Polyline2D]]:
    """Split a lane polygon outline into left and right boundary polylines using the centerline.

    Each polygon vertex is assigned to the left or right side from the sign of the perpendicular dot
    product with the local centerline direction. Returns ``(None, None)`` for any degenerate input
    (too few vertices, or an ambiguous left/right split) so callers can skip the lane rather than crash.

    :param polygon_xy: ``(N, 2)`` array of the lane polygon outline, without the closing duplicate vertex.
    :param centerline_xy: ``(M, 2)`` array of the lane centerline.
    :return: ``(left_boundary, right_boundary)`` as :class:`Polyline2D`, or ``(None, None)`` if degenerate.
    """
    left_boundary: Optional[Polyline2D] = None
    right_boundary: Optional[Polyline2D] = None

    if len(polygon_xy) >= 3 and len(centerline_xy) >= 2:
        # Closest centerline point to each polygon vertex.
        closest_center_idx = np.argmin(cdist(polygon_xy, centerline_xy), axis=1)

        # Local centerline direction at each centerline point.
        direction_vectors = np.diff(
            centerline_xy,
            axis=0,
            prepend=centerline_xy[[0]] - (centerline_xy[[1]] - centerline_xy[[0]]),
        )
        local_dir_vecs = direction_vectors[closest_center_idx]
        origin_to_polygon_vecs = polygon_xy - centerline_xy[closest_center_idx]

        # Perpendicular dot product; < 0 means the vertex is on the right edge of the lane.
        perp_dot_product = (
            local_dir_vecs[:, 0] * origin_to_polygon_vecs[:, 1] - local_dir_vecs[:, 1] * origin_to_polygon_vecs[:, 0]
        )
        on_right = perp_dot_product < 0

        # The left/right vertices form contiguous blocks; locate the single left->right transition.
        # A 0 (no points on one side) or >1 (interleaved) transition count means we cannot split cleanly.
        idx_changes = np.where(np.roll(on_right, 1) < on_right)[0]
        if len(idx_changes) == 1:
            idx = idx_changes.item()
            if idx > 0:
                # Roll so the left/right boundary sits at index 0, ordering points without jumps.
                polygon_xy = np.roll(polygon_xy, shift=-idx, axis=0)
                on_right = np.roll(on_right, shift=-idx)

            left_array = polygon_xy[~on_right]
            right_array = polygon_xy[on_right]

            if len(left_array) > 1 and len(right_array) > 1:
                # Ensure the two edges join into a polygon without their endpoints crossing.
                if endpoints_intersect(left_array, right_array):
                    if not order_matches(left_array, centerline_xy):
                        left_array = left_array[::-1]
                    else:
                        right_array = right_array[::-1]

                left_boundary = Polyline2D.from_array(left_array)
                right_boundary = Polyline2D.from_array(right_array)

    return left_boundary, right_boundary


def endpoints_intersect(left_edge: np.ndarray, right_edge: np.ndarray) -> bool:
    """Check if the segment connecting the endpoints of ``left_edge`` intersects the segment connecting
    the endpoints of ``right_edge``, using the counter-clockwise (CCW) orientation test.
    """

    # NOTE: Code adapted from trajdata, Apache 2.0 License.
    # https://github.com/NVlabs/trajdata/blob/main/src/trajdata/utils/map_utils.py#L177
    def ccw(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> bool:
        return (point_c[1] - point_a[1]) * (point_b[0] - point_a[0]) > (point_b[1] - point_a[1]) * (
            point_c[0] - point_a[0]
        )

    a, b = left_edge[-1], right_edge[-1]
    c, d = right_edge[0], left_edge[0]
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def order_matches(points: np.ndarray, reference: np.ndarray) -> bool:
    """Check if ``points`` and ``reference`` have the same ordering direction, by comparing the distance
    of the start and end of ``points`` to the start of ``reference``.
    """
    # NOTE: Code adapted from trajdata, Apache 2.0 License.
    # https://github.com/NVlabs/trajdata/blob/main/src/trajdata/utils/map_utils.py#L162
    return bool(np.linalg.norm(points[0] - reference[0]) <= np.linalg.norm(points[-1] - reference[0]))


def order_lanes_left_to_right(polylines: List[Polyline2D]) -> List[int]:
    """Order lanes from left to right based on their position.

    :param polylines: List of polylines representing lane centerlines.
    :return: List of indices into ``polylines`` ordered from left (first) to right (last).
    """
    ordered_indices: List[int] = []
    if len(polylines) > 0:
        # Step 1: Compute the average direction vector across all lanes.
        all_directions = []
        for polyline in polylines:
            polyline_array = polyline.array
            if len(polyline_array) >= 2:
                all_directions.append(np.array(polyline_array[-1]) - np.array(polyline_array[0]))

        avg_direction = np.mean(all_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)

        # Step 2: Compute the perpendicular (left) vector by rotating 90 degrees counter-clockwise.
        left_vector = np.array([-avg_direction[1], avg_direction[0]])

        # Step 3: For each lane, project the midpoint of its endpoints onto the left vector.
        lane_positions = []
        for index, polyline in enumerate(polylines):
            if len(polyline) == 0:
                lane_positions.append((index, 0.0))
            else:
                midpoint = (np.array(polyline[0]) + np.array(polyline[-1])) / 2.0
                lane_positions.append((index, float(np.dot(midpoint, left_vector))))

        # Step 4: Sort by projection (higher values are more to the left).
        lane_positions.sort(key=lambda item: item[1], reverse=True)
        ordered_indices = [index for index, _ in lane_positions]

    return ordered_indices
