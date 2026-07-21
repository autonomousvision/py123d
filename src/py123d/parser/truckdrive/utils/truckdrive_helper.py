"""Shared helpers for TruckDrive dataset parsing."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from py123d.geometry import PoseSE3

logger = logging.getLogger(__name__)

_SYNCED_STEM_RE = re.compile(r"^(?P<sync>\d+)_(?P<ts>\d+)$", re.ASCII)


@dataclass(frozen=True)
class SyncedFile:
    """A synchronized TruckDrive file entry."""

    sync_id: int
    timestamp_ns: int
    filename: str


@dataclass(frozen=True)
class TrajectoryPose:
    """One row from ``poses/gt_trajectory.txt``."""

    sync_key: int
    timestamp_s: float
    pose: PoseSE3


def parse_synced_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse ``<sync_id>_<timestamp_ns>.<ext>`` filenames."""
    stem = Path(filename).stem
    match = _SYNCED_STEM_RE.match(stem)
    if match is None:
        return None, None
    return int(match.group("sync")), int(match.group("ts"))


def iter_synced_files(folder: Path, extensions: Sequence[str]) -> Iterator[SyncedFile]:
    """Yield synchronized files from a directory."""
    if not folder.is_dir():
        return

    ext_set = {extension.lower().lstrip(".") for extension in extensions}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        sync_id, timestamp_ns = parse_synced_filename(path.name)
        if sync_id is None or timestamp_ns is None:
            continue
        if path.suffix.lower().lstrip(".") not in ext_set:
            continue
        yield SyncedFile(sync_id=sync_id, timestamp_ns=timestamp_ns, filename=path.name)


def build_sync_file_index(folder: Path, extensions: Sequence[str]) -> Dict[int, SyncedFile]:
    """Build a sync-id keyed index, preferring the smallest timestamp on duplicates."""
    index: Dict[int, SyncedFile] = {}
    for synced_file in iter_synced_files(folder, extensions):
        existing = index.get(synced_file.sync_id)
        if existing is None or synced_file.timestamp_ns < existing.timestamp_ns:
            index[synced_file.sync_id] = synced_file
    return index


def discover_sync_ids(scene_dir: Path) -> List[int]:
    """Discover all sync ids present across core sensor modalities."""
    sync_ids: set[int] = set()

    camera_root = scene_dir / "camera" / "leopard"
    if camera_root.is_dir():
        for camera_dir in camera_root.iterdir():
            images_dir = camera_dir / "images"
            sync_ids.update(build_sync_file_index(images_dir, ["jpg", "jpeg"]).keys())

    aeva_dir = scene_dir / "lidar" / "aeva" / "joint_lidars" / "points"
    sync_ids.update(build_sync_file_index(aeva_dir, ["bin"]).keys())

    ouster_root = scene_dir / "lidar" / "ouster"
    if ouster_root.is_dir():
        for ouster_dir in ouster_root.iterdir():
            points_dir = ouster_dir / "points"
            sync_ids.update(build_sync_file_index(points_dir, ["bin"]).keys())

    return sorted(sync_ids)


def load_tf_tree(tf_tree_path: Path) -> Dict[str, Any]:
    """Load the TruckDrive calibration TF tree JSON."""
    with tf_tree_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dict_to_4x4(transform_dict: Dict[str, Any]) -> np.ndarray:
    """Convert a TruckDrive transform dict to a 4x4 matrix."""
    from pyquaternion import Quaternion

    rotation = transform_dict["rotation"]
    quaternion = Quaternion(
        w=rotation["w"],
        x=rotation["x"],
        y=rotation["y"],
        z=rotation["z"],
    )
    translation = np.array(
        [
            transform_dict["translation"]["x"],
            transform_dict["translation"]["y"],
            transform_dict["translation"]["z"],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion.rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def build_graph_of_transforms(data_extrinsics: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Build a bidirectional transform graph from the TruckDrive TF tree."""
    graph: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for value in data_extrinsics.values():
        parent = value["header"]["frame_id"]
        child = value["child_frame_id"]
        parent_to_child = dict_to_4x4(value["transform"])
        graph[parent][child] = np.linalg.inv(parent_to_child)
        graph[child][parent] = parent_to_child
    return graph


def find_transform(graph: Dict[str, Dict[str, np.ndarray]], src_node: str, tgt_node: str) -> np.ndarray:
    """Find ``T_src_to_tgt`` via BFS over the transform graph."""
    if src_node == tgt_node:
        return np.eye(4, dtype=np.float64)
    if src_node not in graph or tgt_node not in graph:
        raise ValueError(f"Frame not found in TF graph: src={src_node}, tgt={tgt_node}")

    visited: set[str] = set()
    queue: deque[Tuple[str, np.ndarray]] = deque([(src_node, np.eye(4, dtype=np.float64))])
    while queue:
        current_frame, transform_src_to_current = queue.popleft()
        if current_frame == tgt_node:
            return transform_src_to_current
        visited.add(current_frame)
        for neighbor_frame, transform_current_to_neighbor in graph[current_frame].items():
            if neighbor_frame in visited:
                continue
            transform_src_to_neighbor = transform_current_to_neighbor @ transform_src_to_current
            queue.append((neighbor_frame, transform_src_to_neighbor))

    raise ValueError(f"No transform path from '{src_node}' to '{tgt_node}'.")


class TransformTree:
    """Cached TruckDrive static transform tree."""

    def __init__(self, tf_tree_path: Path) -> None:
        self._graph = build_graph_of_transforms(load_tf_tree(tf_tree_path))

    def lookup(self, src_node: str, tgt_node: str) -> PoseSE3:
        """Return ``T_src_to_tgt`` as a :class:`PoseSE3`."""
        matrix = find_transform(self._graph, src_node, tgt_node)
        return PoseSE3.from_transformation_matrix(matrix)


def load_gt_trajectory(trajectory_path: Path) -> Dict[int, TrajectoryPose]:
    """Load ``poses/gt_trajectory.txt`` keyed by sync key."""
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Missing trajectory file: {trajectory_path}")

    poses: Dict[int, TrajectoryPose] = {}
    with trajectory_path.open("r", encoding="utf-8") as file:
        header = file.readline()
        if not header:
            return poses
        for line in file:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            if parts[0] == "refined_gt_trajectory":
                continue
            sync_key = int(float(parts[0]))
            timestamp_s = float(parts[1])
            x, y, z = map(float, parts[2:5])
            qx, qy, qz, qw = map(float, parts[5:9])
            pose = PoseSE3(x=x, y=y, z=z, qw=qw, qx=qx, qy=qy, qz=qz)
            poses[sync_key] = TrajectoryPose(sync_key=sync_key, timestamp_s=timestamp_s, pose=pose)
    return poses


def discover_camera_names(scene_dir: Path) -> List[str]:
    """Discover leopard camera names present on disk for a scene."""
    camera_root = scene_dir / "camera" / "leopard"
    if not camera_root.is_dir():
        return []
    return sorted(path.name for path in camera_root.iterdir() if path.is_dir())


def discover_ouster_names(scene_dir: Path) -> List[str]:
    """Discover ouster lidar names present on disk for a scene."""
    ouster_root = scene_dir / "lidar" / "ouster"
    if not ouster_root.is_dir():
        return []
    return sorted(path.name for path in ouster_root.iterdir() if path.is_dir())


def load_camera_calibration(calib_path: Path) -> Tuple[int, int, float, float, float, float]:
    """Load width, height, fx, fy, cx, cy from a TruckDrive camera calibration JSON."""
    with calib_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    k = data["K"]
    return int(data["height"]), int(data["width"]), float(k[0]), float(k[4]), float(k[2]), float(k[5])


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    if points.size == 0:
        return points.reshape((0, 3))
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]
