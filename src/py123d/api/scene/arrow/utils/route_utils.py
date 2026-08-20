"""Driven-route polyline: computation, ``route.arrow`` I/O, and interpolation.

The route polyline is the ego's driven path over the whole log, pruned of
standstill jitter and resampled at a fixed arc-length resolution. It is written
once per log by :class:`~py123d.api.scene.arrow.arrow_log_writer.ArrowLogWriter`
alongside a per-frame ``sync.route_progress_m`` column holding each frame's
arc-length position on the polyline.
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
"""Sync-table column holding each frame's arc-length position on the route polyline in meters."""

_ROUTE_ARC_COLUMN = "route.arc_m"
_ROUTE_XYZ_COLUMNS = ("route.x", "route.y", "route.z")

_MIN_SEGMENT_M = 0.05
"""Consecutive positions closer than this are pruned so standstill odometry jitter
does not accumulate arc-length."""


@dataclass(frozen=True)
class RouteData:
    """A log's driven route and the per-frame positions on it.

    :param polyline_arc_m: Arc-length of each polyline vertex, shape (K,), starting at 0.
    :param polyline_xyz: Polyline vertices in the ego odometry frame, shape (K, 3).
    :param progress_m: Arc-length position of each input frame on the polyline, shape (N,),
        monotone non-decreasing.
    :param total_arc_m: Total arc-length of the driven route in meters.
    """

    polyline_arc_m: npt.NDArray[np.float64]
    polyline_xyz: npt.NDArray[np.float64]
    progress_m: npt.NDArray[np.float64]
    total_arc_m: float


def compute_route_data(
    positions_xyz: npt.NDArray[np.float64],
    resolution_m: float,
    min_segment_m: float = _MIN_SEGMENT_M,
) -> Optional[RouteData]:
    """Compute the driven-route polyline and per-frame progress from ego positions.

    Positions are pruned so consecutive kept vertices are at least ``min_segment_m``
    apart: standstill jitter then contributes no arc-length, and a frame's progress is
    the arc-length of the last kept vertex at or before it (piecewise-constant while
    standing, exact while driving).

    :param positions_xyz: Ego positions over the whole log, shape (N, 3), in log order.
    :param resolution_m: Arc-length spacing of the resampled polyline vertices.
    :param min_segment_m: Minimum distance between kept vertices before resampling.
    :return: The route data, or None when no position is available.
    """
    assert resolution_m > 0.0, f"resolution_m must be > 0, got {resolution_m}."
    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    num_positions = positions_xyz.shape[0]
    if num_positions == 0:
        return None
    assert positions_xyz.ndim == 2 and positions_xyz.shape[1] == 3, (
        f"Expected positions of shape (N, 3), got {positions_xyz.shape}."
    )

    # Prune: keep a position once it moved min_segment_m away from the last kept one.
    kept_indices = [0]
    last_kept = positions_xyz[0]
    for index in range(1, num_positions):
        if float(np.linalg.norm(positions_xyz[index] - last_kept)) >= min_segment_m:
            kept_indices.append(index)
            last_kept = positions_xyz[index]
    kept_index_array = np.asarray(kept_indices, dtype=np.int64)
    kept_xyz = positions_xyz[kept_index_array]

    # Arc-length parameterization of the pruned path; segments are >= min_segment_m,
    # so the parameterization is strictly increasing.
    segment_lengths = np.linalg.norm(np.diff(kept_xyz, axis=0), axis=1)
    kept_arc = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64)))
    total_arc_m = float(kept_arc[-1])

    # Per-frame progress: arc-length of the last kept vertex at or before the frame.
    frame_positions = np.searchsorted(kept_index_array, np.arange(num_positions, dtype=np.int64), side="right") - 1
    progress_m = kept_arc[frame_positions]

    # Resample at fixed resolution; the final vertex always lands exactly on total_arc_m.
    sample_arc = np.arange(0.0, total_arc_m, resolution_m, dtype=np.float64)
    if len(sample_arc) == 0 or total_arc_m - sample_arc[-1] > 1e-9:
        sample_arc = np.concatenate((sample_arc, [total_arc_m]))
    polyline_xyz = np.stack(
        [np.interp(sample_arc, kept_arc, kept_xyz[:, axis]) for axis in range(3)],
        axis=1,
    )

    return RouteData(
        polyline_arc_m=sample_arc,
        polyline_xyz=polyline_xyz,
        progress_m=progress_m,
        total_arc_m=total_arc_m,
    )


def interpolate_route_at_arc(
    polyline_arc_m: npt.NDArray[np.float64],
    polyline_xyz: npt.NDArray[np.float64],
    query_arc_m: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolate route positions at the given arc-length values.

    Queries outside ``[0, total_arc_m]`` are clamped to the polyline's endpoints;
    callers that must not read past the route's end should check the query against
    the total arc-length first.

    :param polyline_arc_m: Arc-length of each polyline vertex, shape (K,).
    :param polyline_xyz: Polyline vertices, shape (K, 3).
    :param query_arc_m: Arc-length values to interpolate at, shape (M,).
    :return: Interpolated positions, shape (M, 3).
    """
    query_arc_m = np.asarray(query_arc_m, dtype=np.float64)
    return np.stack(
        [np.interp(query_arc_m, polyline_arc_m, polyline_xyz[:, axis]) for axis in range(3)],
        axis=1,
    )


def write_route_arrow(
    log_dir: Path,
    route_data: RouteData,
    route_metadata: RouteMetadata,
    ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
    ipc_compression_level: Optional[int] = None,
) -> None:
    """Write a log's route polyline to ``route.arrow``.

    :param log_dir: The log directory.
    :param route_data: The computed route data.
    :param route_metadata: The route metadata, stored in the file's schema metadata.
    :param ipc_compression: The IPC compression method, defaults to None.
    :param ipc_compression_level: The IPC compression level, defaults to None.
    """
    schema = pa.schema(
        [
            pa.field(_ROUTE_ARC_COLUMN, pa.float64()),
            pa.field(_ROUTE_XYZ_COLUMNS[0], pa.float64()),
            pa.field(_ROUTE_XYZ_COLUMNS[1], pa.float64()),
            pa.field(_ROUTE_XYZ_COLUMNS[2], pa.float64()),
        ]
    )
    schema = add_metadata_to_arrow_schema(schema, route_metadata)

    table = pa.table(
        {
            _ROUTE_ARC_COLUMN: route_data.polyline_arc_m,
            _ROUTE_XYZ_COLUMNS[0]: route_data.polyline_xyz[:, 0],
            _ROUTE_XYZ_COLUMNS[1]: route_data.polyline_xyz[:, 1],
            _ROUTE_XYZ_COLUMNS[2]: route_data.polyline_xyz[:, 2],
        },
        schema=schema,
    )

    options = None
    if ipc_compression is not None:
        options = pa.ipc.IpcWriteOptions(
            compression=pa.Codec(ipc_compression, compression_level=ipc_compression_level)
        )

    source = pa.OSFile(str(log_dir / ROUTE_FILE_NAME), "wb")
    writer = pa.ipc.new_file(source, schema=schema, options=options)
    try:
        writer.write_table(table)
    finally:
        writer.close()
        source.close()


def read_route_arrow(log_dir: Path) -> Optional[Tuple[RouteMetadata, npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Read a log's route polyline from ``route.arrow``.

    :param log_dir: The log directory.
    :return: Tuple of (route metadata, arc-length per vertex (K,), vertices (K, 3)),
        or None when the log has no route file.
    """
    from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table

    route_path = Path(log_dir) / ROUTE_FILE_NAME
    if not route_path.exists():
        return None

    table = get_lru_cached_arrow_table(str(route_path))
    route_metadata = get_metadata_from_arrow_schema(table.schema, RouteMetadata)
    polyline_arc_m = table[_ROUTE_ARC_COLUMN].to_numpy()
    polyline_xyz = np.stack([table[column].to_numpy() for column in _ROUTE_XYZ_COLUMNS], axis=1)
    return route_metadata, polyline_arc_m, polyline_xyz
