"""Route polyline: computation, ``route.arrow`` I/O, and interpolation.

A log's route is one polyline resampled at a fixed arc-length resolution — the driven
path derived from ego odometry, or an explicitly provided (planned) route. The log
writer stores it in ``route.arrow`` and each frame's arc-length position on it in the
``sync.route_progress_m`` column.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema, get_metadata_from_arrow_schema
from py123d.datatypes.metadata.route_metadata import RouteMetadata

ROUTE_FILE_NAME = "route.arrow"

SYNC_ROUTE_PROGRESS_COLUMN = "sync.route_progress_m"
"""Sync-table column: each frame's arc-length position on the route polyline in meters."""

_ROUTE_ARC_COLUMN = "route.arc_m"
_ROUTE_XYZ_COLUMNS = ("route.x", "route.y", "route.z")

_MIN_SEGMENT_M = 0.05
"""Positions closer than this to the previously kept one are pruned, so standstill
odometry jitter accumulates no arc-length."""

_PROJECTION_WINDOW_M = 50.0
"""Forward search window when projecting ego positions onto a provided route; keeps the
projection monotone even where the route crosses itself."""


@dataclass(frozen=True)
class RouteData:
    """A log's route polyline and the per-frame positions on it.

    ``polyline_arc_m`` (K,) holds the arc-length of each vertex in ``polyline_xyz`` (K, 3);
    ``progress_m`` (N,) is each ego frame's monotone arc-length position, or None when the
    log has no ego frames to attach progress to.
    """

    polyline_arc_m: npt.NDArray[np.float64]
    polyline_xyz: npt.NDArray[np.float64]
    progress_m: Optional[npt.NDArray[np.float64]]
    total_arc_m: float


def _resampled_polyline(
    positions_xyz: npt.NDArray[np.float64], resolution_m: float, min_segment_m: float
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Prune positions within ``min_segment_m`` of the previously kept one (strictly
    increasing arc parameterization), then resample at ``resolution_m`` steps with the
    final vertex landing exactly on the total arc.

    :return: (kept indices, arc per kept position, resampled arc, resampled vertices).
    """
    kept = [0]
    for index in range(1, len(positions_xyz)):
        if float(np.linalg.norm(positions_xyz[index] - positions_xyz[kept[-1]])) >= min_segment_m:
            kept.append(index)
    kept_indices = np.asarray(kept, dtype=np.int64)
    kept_xyz = positions_xyz[kept_indices]
    kept_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(kept_xyz, axis=0), axis=1), dtype=np.float64)))

    total_arc_m = float(kept_arc[-1])
    sample_arc = np.arange(0.0, total_arc_m, resolution_m, dtype=np.float64)
    if len(sample_arc) == 0 or total_arc_m - sample_arc[-1] > 1e-9:
        sample_arc = np.concatenate((sample_arc, [total_arc_m]))
    return kept_indices, kept_arc, sample_arc, interpolate_route_at_arc(kept_arc, kept_xyz, sample_arc)


def _as_positions(positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    assert positions_xyz.ndim == 2 and positions_xyz.shape[1] == 3, (
        f"Expected positions of shape (N, 3), got {positions_xyz.shape}."
    )
    return positions_xyz


def compute_route_data(
    positions_xyz: npt.NDArray[np.float64],
    resolution_m: float,
    min_segment_m: float = _MIN_SEGMENT_M,
) -> Optional[RouteData]:
    """Compute the driven route and per-frame progress from ego positions (N, 3) in log order.

    A frame's progress is the arc-length of the last kept (pruned) position at or before
    it: piecewise-constant while standing, exact while driving. Returns None when no
    position is available.
    """
    assert resolution_m > 0.0, f"resolution_m must be > 0, got {resolution_m}."
    positions_xyz = _as_positions(positions_xyz)
    if len(positions_xyz) == 0:
        return None

    kept_indices, kept_arc, sample_arc, sample_xyz = _resampled_polyline(positions_xyz, resolution_m, min_segment_m)
    frame_positions = np.searchsorted(kept_indices, np.arange(len(positions_xyz), dtype=np.int64), side="right") - 1
    return RouteData(sample_arc, sample_xyz, kept_arc[frame_positions], float(kept_arc[-1]))


def compute_route_data_from_waypoints(
    route_xyz: npt.NDArray[np.float64],
    ego_positions_xyz: Optional[npt.NDArray[np.float64]],
    resolution_m: float,
    min_segment_m: float = _MIN_SEGMENT_M,
) -> Optional[RouteData]:
    """Compute route data from explicitly provided waypoints (M, 3) in driving order.

    Per-frame progress is the ego's monotone projection onto the route, well-defined even
    where the ego deviates from the route or the route crosses itself; pass
    ``ego_positions_xyz=None`` to skip it. Returns None when no waypoint is available.
    """
    assert resolution_m > 0.0, f"resolution_m must be > 0, got {resolution_m}."
    route_xyz = _as_positions(route_xyz)
    if len(route_xyz) == 0:
        return None

    _, kept_arc, sample_arc, sample_xyz = _resampled_polyline(route_xyz, resolution_m, min_segment_m)
    progress_m: Optional[npt.NDArray[np.float64]] = None
    if ego_positions_xyz is not None and len(ego_positions_xyz) > 0:
        progress_m = project_onto_polyline(sample_arc, sample_xyz, _as_positions(ego_positions_xyz))
    return RouteData(sample_arc, sample_xyz, progress_m, float(kept_arc[-1]))


def project_onto_polyline(
    polyline_arc_m: npt.NDArray[np.float64],
    polyline_xyz: npt.NDArray[np.float64],
    positions_xyz: npt.NDArray[np.float64],
    window_m: float = _PROJECTION_WINDOW_M,
) -> npt.NDArray[np.float64]:
    """Project positions (N, 3, temporal order) onto a polyline as monotone arc-length values.

    The first position searches the whole polyline; each subsequent one only from the
    previous match forward by ``window_m``, which keeps progress monotone on self-crossing
    routes.
    """
    progress = np.empty(len(positions_xyz), dtype=np.float64)
    start = 0
    for index, position in enumerate(positions_xyz):
        end = int(np.searchsorted(polyline_arc_m, polyline_arc_m[start] + window_m, side="right"))
        distances = np.linalg.norm(polyline_xyz[start : end + 1] - position, axis=1)
        nearest = start + int(np.argmin(distances))
        best_arc, best_dist = float(polyline_arc_m[nearest]), float(distances[nearest - start])
        # Exact projection onto the two segments adjacent to the nearest vertex.
        for a in (nearest - 1, nearest):
            if a < 0 or a + 1 >= len(polyline_xyz):
                continue
            segment = polyline_xyz[a + 1] - polyline_xyz[a]
            length_sq = float(segment @ segment)
            if length_sq == 0.0:  # out-and-back turnaround can coincide spatially
                continue
            t = float(np.clip((position - polyline_xyz[a]) @ segment / length_sq, 0.0, 1.0))
            dist = float(np.linalg.norm(polyline_xyz[a] + t * segment - position))
            if dist < best_dist:
                best_arc = float(polyline_arc_m[a] + t * (polyline_arc_m[a + 1] - polyline_arc_m[a]))
                best_dist = dist
        progress[index] = max(best_arc, progress[index - 1]) if index > 0 else best_arc
        start = max(int(np.searchsorted(polyline_arc_m, progress[index], side="right")) - 1, 0)
    return progress


def interpolate_route_at_arc(
    polyline_arc_m: npt.NDArray[np.float64],
    polyline_xyz: npt.NDArray[np.float64],
    query_arc_m: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolate route positions (M, 3) at the given arc-length values (M,).

    Queries outside ``[0, total_arc_m]`` clamp to the endpoints; callers that must not
    read past the route's end check the query against the total arc-length first.
    """
    query_arc_m = np.asarray(query_arc_m, dtype=np.float64)
    return np.stack([np.interp(query_arc_m, polyline_arc_m, polyline_xyz[:, axis]) for axis in range(3)], axis=1)


def write_route_arrow(
    log_dir: Path,
    route_data: RouteData,
    route_metadata: RouteMetadata,
    ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
    ipc_compression_level: Optional[int] = None,
) -> None:
    """Write a log's route polyline to ``route.arrow``, metadata in the schema metadata."""
    schema = pa.schema([pa.field(name, pa.float64()) for name in (_ROUTE_ARC_COLUMN, *_ROUTE_XYZ_COLUMNS)])
    schema = add_metadata_to_arrow_schema(schema, route_metadata)
    table = pa.table(
        {
            _ROUTE_ARC_COLUMN: route_data.polyline_arc_m,
            **{name: route_data.polyline_xyz[:, axis] for axis, name in enumerate(_ROUTE_XYZ_COLUMNS)},
        },
        schema=schema,
    )

    options = None
    if ipc_compression is not None:
        options = pa.ipc.IpcWriteOptions(compression=pa.Codec(ipc_compression, compression_level=ipc_compression_level))
    with pa.OSFile(str(log_dir / ROUTE_FILE_NAME), "wb") as source:
        with pa.ipc.new_file(source, schema=schema, options=options) as writer:
            writer.write_table(table)


def read_route_arrow(log_dir: Path) -> Optional[Tuple[RouteMetadata, npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Read (route metadata, arc per vertex (K,), vertices (K, 3)) from ``route.arrow``,
    or None when the log has no route file."""
    from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table

    route_path = Path(log_dir) / ROUTE_FILE_NAME
    if not route_path.exists():
        return None

    table = get_lru_cached_arrow_table(str(route_path))
    route_metadata = get_metadata_from_arrow_schema(table.schema, RouteMetadata)
    polyline_xyz = np.stack([table[column].to_numpy() for column in _ROUTE_XYZ_COLUMNS], axis=1)
    return route_metadata, table[_ROUTE_ARC_COLUMN].to_numpy(), polyline_xyz
