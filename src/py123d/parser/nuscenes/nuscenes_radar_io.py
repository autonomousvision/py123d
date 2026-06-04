"""Point cloud loader for nuScenes radar.

nuScenes ships five independent radars (RADAR_FRONT, RADAR_FRONT_LEFT, RADAR_FRONT_RIGHT,
RADAR_BACK_LEFT, RADAR_BACK_RIGHT), each stored in its own ``.pcd`` file with the rich 18-field
schema (``x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms
invalid_state pdh0 vx_rms vy_rms``). We follow the merged model: each radar's returns are read,
mapped to :class:`~py123d.datatypes.sensors.radar.RadarFeature`, transformed sensor→ego, tagged
with the originating :class:`~py123d.datatypes.sensors.radar.RadarID`, and concatenated into one
cloud (splittable back per-sensor at API read time).

The per-radar relative paths are carried through :class:`~py123d.parser.base_dataset_parser.ParsedRadar`'s
``load_kwargs`` (under ``"radar_paths"``), since the merged frame stores only a single primary path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from py123d.datatypes.sensors.radar import RADAR_FEATURE_DTYPES, RadarFeature, RadarID, RadarMetadata
from py123d.geometry import PoseSE3
from py123d.geometry.transform import reframe_points_3d_array

# nuScenes radar PCD column index -> RadarFeature (columns 0..2 are x, y, z, handled separately).
_NUSCENES_RADAR_COLUMN_TO_FEATURE: Dict[int, RadarFeature] = {
    3: RadarFeature.DYN_PROP,
    4: RadarFeature.CLUSTER_ID,
    5: RadarFeature.RCS,
    6: RadarFeature.VELOCITY_X,
    7: RadarFeature.VELOCITY_Y,
    8: RadarFeature.VELOCITY_X_COMP,
    9: RadarFeature.VELOCITY_Y_COMP,
    10: RadarFeature.IS_QUALITY_VALID,
    11: RadarFeature.AMBIG_STATE,
    12: RadarFeature.X_RMS,
    13: RadarFeature.Y_RMS,
    14: RadarFeature.INVALID_STATE,
    15: RadarFeature.PDH0,
    16: RadarFeature.VX_RMS,
    17: RadarFeature.VY_RMS,
}


def load_nuscenes_radar_point_cloud_data_from_path(
    pcd_path: Path,
    radar_metadatas: Dict[RadarID, RadarMetadata],
    sensor_root: Path,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads and merges the nuScenes radar point clouds for one keyframe into the ego frame.

    :param pcd_path: Absolute path to the primary radar ``.pcd`` (unused beyond existence; the full
        per-radar path set comes from ``load_kwargs["radar_paths"]``).
    :param radar_metadatas: Per-:class:`RadarID` metadata (provides the sensor→ego extrinsics).
    :param sensor_root: Dataset sensor root, against which the relative radar paths are resolved.
    :param load_kwargs: Must carry ``"radar_paths"``: a ``{str(int(RadarID)): relative_path}`` mapping.
    :return: Tuple of (Nx3 float32 xyz in ego frame, features dict keyed by ``RadarFeature.serialize()``).
    """
    from nuscenes.utils.data_classes import RadarPointCloud

    kw = load_kwargs or {}
    radar_paths: Dict[str, str] = kw.get("radar_paths", {})
    assert radar_paths, "nuScenes radar loading requires 'radar_paths' in load_kwargs."

    xyz_list: List[np.ndarray] = []
    features_list: Dict[str, List[np.ndarray]] = {}

    for radar_id_str, relative_path in radar_paths.items():
        radar_id = RadarID(int(radar_id_str))
        metadata = radar_metadatas.get(radar_id)
        if metadata is None:
            continue
        full_path = Path(sensor_root) / relative_path
        if not full_path.exists():
            continue

        # Keep every return (do not drop invalid/ambiguous points) — this is an archival representation.
        radar_pc = RadarPointCloud.from_file(
            str(full_path),
            invalid_states=list(range(18)),
            dynprop_states=list(range(8)),
            ambig_states=list(range(5)),
        )
        points = radar_pc.points  # shape (18, N)
        num_points = points.shape[1]
        if num_points == 0:
            continue

        extrinsic = metadata.radar_to_imu_se3
        # Positions: sensor frame -> ego/IMU frame.
        xyz_ego = reframe_points_3d_array(
            from_origin=extrinsic,
            to_origin=PoseSE3.identity(),
            points_3d_array=points[:3, :].T.astype(np.float32),  # type: ignore
        ).astype(np.float32)
        xyz_list.append(xyz_ego)

        # Per-point features.
        ids = np.full(num_points, int(radar_id), dtype=RADAR_FEATURE_DTYPES[RadarFeature.IDS])
        _append_feature(features_list, RadarFeature.IDS, ids)
        for col, feature in _NUSCENES_RADAR_COLUMN_TO_FEATURE.items():
            _append_feature(features_list, feature, points[col, :].astype(RADAR_FEATURE_DTYPES[feature]))

        # Velocities are free vectors in the sensor frame; rotate (no translation) into the ego frame.
        rotation = extrinsic.rotation_matrix
        _rotate_velocity_into_ego(features_list, RadarFeature.VELOCITY_X, RadarFeature.VELOCITY_Y, rotation, num_points)
        _rotate_velocity_into_ego(
            features_list, RadarFeature.VELOCITY_X_COMP, RadarFeature.VELOCITY_Y_COMP, rotation, num_points
        )

    if not xyz_list:
        return np.zeros((0, 3), dtype=np.float32), {}

    point_cloud_3d = np.concatenate(xyz_list, axis=0)
    point_cloud_features = {key: np.concatenate(values, axis=0) for key, values in features_list.items()}
    return point_cloud_3d, point_cloud_features


def _append_feature(features_list: Dict[str, List[np.ndarray]], feature: RadarFeature, values: np.ndarray) -> None:
    key = feature.serialize()
    features_list.setdefault(key, []).append(values)


def _rotate_velocity_into_ego(
    features_list: Dict[str, List[np.ndarray]],
    vx_feature: RadarFeature,
    vy_feature: RadarFeature,
    rotation: np.ndarray,
    num_points: int,
) -> None:
    """Rotates the most-recently appended (vx, vy) arrays for one radar into the ego frame, in place."""
    vx = features_list[vx_feature.serialize()][-1]
    vy = features_list[vy_feature.serialize()][-1]
    velocity_sensor = np.stack([vx, vy, np.zeros(num_points, dtype=np.float32)], axis=0)  # (3, N)
    velocity_ego = (rotation @ velocity_sensor).astype(np.float32)
    features_list[vx_feature.serialize()][-1] = velocity_ego[0]
    features_list[vy_feature.serialize()][-1] = velocity_ego[1]
