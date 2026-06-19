"""Lazy LiDAR loader for Griffin ``.ply`` point clouds.

Invoked from
:func:`py123d.common.io.lidar.path_lidar_io.load_point_cloud_data_from_path`
when a stored ``ParsedLidar`` is materialized (during conversion with the
``binary`` codec, or at read time with the ``path`` codec).

Griffin specifics handled here:

- Points are stored in the **ego frame** (vehicle at the origin, X-forward,
  Y-left, Z-up), already merged from the single top LiDAR, so they are returned
  unchanged - no pose or extrinsic transform is applied.
- The intensity field is the **uppercase** ``I`` (a known gotcha: many tools
  expect lowercase ``intensity``). Candidate names are probed in priority order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import LidarFeature, LidarID

# Candidate intensity property names, in priority order (Griffin uses ``I``).
_INTENSITY_FIELDS: Tuple[str, ...] = ("I", "intensity", "i")


def load_griffin_point_cloud_data_from_path(
    ply_path: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load a Griffin ego-frame LiDAR ``.ply`` into points and features.

    :param ply_path: Absolute path to the ``{frame}.ply`` LiDAR file.
    :return: Tuple of (point_cloud_3d ``[N, 3]`` float32 in the ego frame,
        features dict with intensity and per-point LiDAR ids).
    """
    check_dependencies(["plyfile"], "griffin")
    from plyfile import PlyData

    ply_path = Path(ply_path)
    assert ply_path.exists(), f"Griffin LiDAR file not found: {ply_path}"

    vertex = PlyData.read(str(ply_path))["vertex"]
    property_names = {prop.name for prop in vertex.properties}

    point_cloud_3d = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=-1,
    )
    num_points = len(point_cloud_3d)

    intensity_field = next((name for name in _INTENSITY_FIELDS if name in property_names), None)
    intensity = (
        np.asarray(vertex[intensity_field], dtype=np.float32)
        if intensity_field is not None
        else np.zeros(num_points, dtype=np.float32)
    )

    lidar_ids = np.full(num_points, int(LidarID.LIDAR_TOP), dtype=np.uint8)

    point_cloud_features: Dict[str, np.ndarray] = {
        LidarFeature.INTENSITY.serialize(): intensity,
        LidarFeature.IDS.serialize(): lidar_ids,
    }
    return point_cloud_3d, point_cloud_features
