"""Low-level helpers for the Griffin parser: scenes, splits, poses, calib, labels.

All geometric conventions follow the 123D unified frame (ISO 8855: X-forward,
Y-left, Z-up; scalar-first quaternions). Griffin's raw conventions and how they
map onto 123D are:

- **Ego frame**: forward-left-up, identical to 123D - no axis remapping needed.
- **Pose** (``pose/{frame}.json``): the ego vehicle in an ENU world frame as
  ``(x, y, z, roll, pitch, yaw)`` with translation in metres and angles in
  **degrees**. The Euler order is ``xyz`` as used by Griffin's ``space_utils.py``
  (SciPy ``Rotation.from_euler('xyz', ...)``). We build the rotation matrix with
  SciPy and hand a 4x4 to :meth:`PoseSE3.from_transformation_matrix`, so no
  Euler-order assumptions leak in. The resulting transform is ego-to-global.
- **Calibration** (``calib/{sensor}.json``): ``extrinsic`` is the 4x4
  **sensor-to-ego** transform; ``intrinsic`` (cameras only) is the 3x3 pinhole
  matrix. Because 123D's ego frame doubles as the IMU frame, the extrinsic *is*
  the ``camera_to_imu_se3`` / ``lidar_to_imu_se3`` directly - no inversion.
- **Labels** (``label/{frame}.txt``) and **LiDAR points** are in the **ego
  frame**. Boxes are lifted to the global frame here (123D stores
  ``BoxDetectionsSE3`` globally); LiDAR points are kept ego-relative (the 123D
  log writer re-expresses them as needed).

References
----------
.. [1] Official toolkit: https://github.com/wang-jh18-SVM/Griffin
   (see ``tools/griffin_data_converter/data_utils.py`` for the raw schema).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.geometry import PoseSE3
from py123d.geometry.transform import rel_to_abs_se3

# Number of fields in a fully-specified Griffin label line.
_FULL_LABEL_FIELDS = 12


def read_json(path: Union[str, Path]) -> Any:
    """Read and parse a JSON file.

    :param path: Path to the JSON file.
    :return: The decoded JSON content.
    """
    with open(path, "r") as f:
        return json.load(f)


def euler_xyz_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from Griffin ``xyz`` Euler angles (degrees).

    Uses SciPy's ``'xyz'`` convention to match Griffin's ``space_utils.py``. The
    optional dependency is imported lazily so importing this module does not
    require ``scipy`` unless a Griffin conversion actually runs.

    :param roll_deg: Rotation about X (forward), in degrees.
    :param pitch_deg: Rotation about Y (left), in degrees.
    :param yaw_deg: Rotation about Z (up), in degrees.
    :return: A ``(3, 3)`` rotation matrix.
    """
    check_dependencies(["scipy"], "griffin")
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


def pose_dict_to_ego_to_global_se3(pose: Dict[str, float]) -> PoseSE3:
    """Convert a Griffin pose dict to an ego-to-global :class:`PoseSE3`.

    :param pose: Dict with keys ``x, y, z, roll, pitch, yaw`` (metres, degrees).
    :return: The ego-to-global pose.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_xyz_to_rotation_matrix(float(pose["roll"]), float(pose["pitch"]), float(pose["yaw"]))
    transform[:3, 3] = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    return PoseSE3.from_transformation_matrix(transform)


def load_calibration(calib_dir: Union[str, Path], sensor: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Load intrinsic/extrinsic calibration for one sensor.

    :param calib_dir: Path to the side's ``calib`` directory.
    :param sensor: Sensor name, e.g. ``"front"`` or ``"lidar_top"``.
    :return: Tuple of (intrinsic ``(3, 3)`` or ``None`` if absent, extrinsic ``(4, 4)``).
        The extrinsic is the raw sensor-to-ego transform.
    """
    data = read_json(Path(calib_dir) / f"{sensor}.json")
    extrinsic = np.array(data["extrinsic"], dtype=np.float64)
    assert extrinsic.shape == (4, 4), f"Griffin extrinsic for '{sensor}' must be 4x4, got {extrinsic.shape}."
    intrinsic = None
    if data.get("intrinsic") is not None:
        intrinsic = np.array(data["intrinsic"], dtype=np.float64)
        assert intrinsic.shape == (3, 3), f"Griffin intrinsic for '{sensor}' must be 3x3, got {intrinsic.shape}."
    return intrinsic, extrinsic


def sensor_to_ego_pose_se3(calib_dir: Union[str, Path], sensor: str) -> PoseSE3:
    """Return the sensor-to-ego (== sensor-to-IMU) pose for ``sensor``.

    :param calib_dir: Path to the side's ``calib`` directory.
    :param sensor: Sensor name, e.g. ``"lidar_top"``.
    :return: The sensor-to-ego pose.
    """
    _, extrinsic = load_calibration(calib_dir, sensor)
    return PoseSE3.from_transformation_matrix(extrinsic)


def load_scene_index(vehicle_root: Union[str, Path]) -> List[Tuple[str, List[str]]]:
    """Read ``scene_infos.json`` and return ordered ``(full_scene_name, frames)`` pairs.

    Each Griffin scene is a contiguous ~150-frame (~15 s at 10 Hz) sequence and
    maps onto one 123D log. The full scene name matches the identifiers used in
    the official split files: ``scene-{idx:04d}-{name}`` where ``idx`` is the
    scene's position in ``scene_infos.json`` and ``name`` is its ``name`` field
    (e.g. ``scene-0026-Town07-001``). This mirrors ``generate_scene_metadata`` in
    the official converter.

    :param vehicle_root: Path to ``vehicle-side``.
    :return: List of ``(full_scene_name, [frame_id, ...])`` in dataset order,
        with frame ids sorted (6-digit zero-padded, so lexicographic == temporal).
    """
    scene_infos = read_json(Path(vehicle_root) / "scene_infos.json")
    scenes: List[Tuple[str, List[str]]] = []
    for idx, scene_info in enumerate(scene_infos):
        info = scene_info["info"]
        name = scene_info["name"]
        full_name = f"scene-{idx:04d}-{name}"
        frames = sorted(str(frame) for frame in info["frames"])
        scenes.append((full_name, frames))
    return scenes


def load_split_scene_names(split_file: Union[str, Path], kind: str) -> List[str]:
    """Read the official split file and return the scene names for one partition.

    :param split_file: Path to ``{subset}.json`` (embedded under
        :mod:`py123d.parser.griffin.splits`).
    :param kind: Partition name, ``"train"`` or ``"val"``.
    :return: Ordered list of full scene names assigned to ``kind``.
    :raises KeyError: If ``kind`` is not present in the split file.
    """
    split_data = read_json(split_file)
    batch_split = split_data["batch_split"]
    if kind not in batch_split:
        raise KeyError(f"Split kind '{kind}' not found in {split_file}. Available: {list(batch_split)}")
    return [str(name) for name in batch_split[kind]]


def parse_label_file(label_path: Union[str, Path]) -> List[Dict]:
    """Parse a Griffin label ``.txt`` into a list of annotation dicts (ego frame).

    Line format (space-separated)::

        type x y z l w h roll pitch yaw id visibility

    Robust to the two abbreviated variants in the release: 11 fields (missing
    ``visibility``) and 9 fields (missing ``roll``, ``pitch`` and ``visibility``),
    matching the official loader's tolerance.

    :param label_path: Path to ``{frame}.txt``.
    :return: List of annotation dicts. Empty if the file is missing or empty.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    annotations: List[Dict] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue

        n = len(parts)
        if n == _FULL_LABEL_FIELDS:
            roll, pitch = float(parts[7]), float(parts[8])
            yaw, track_id, visibility = float(parts[9]), parts[10], float(parts[11])
        elif n == _FULL_LABEL_FIELDS - 1:  # missing visibility
            roll, pitch = float(parts[7]), float(parts[8])
            yaw, track_id, visibility = float(parts[9]), parts[10], 1.0
        elif n == _FULL_LABEL_FIELDS - 3:  # missing roll, pitch, visibility
            roll, pitch = 0.0, 0.0
            yaw, track_id, visibility = float(parts[7]), parts[8], 1.0
        else:
            continue

        annotations.append(
            {
                "type": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "l": float(parts[4]),
                "w": float(parts[5]),
                "h": float(parts[6]),
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "track_id": str(track_id),
                "visibility": visibility,
            }
        )
    return annotations


def ego_box_to_global_se3(annotation: Dict, ego_to_global: PoseSE3) -> PoseSE3:
    """Lift an ego-frame box pose into the global frame.

    The box centre and orientation are given in the ego frame (Euler ``xyz``,
    degrees); composing with the ego-to-global pose yields the global box pose
    that 123D's ``BoundingBoxSE3`` expects.

    :param annotation: Annotation dict from :func:`parse_label_file`.
    :param ego_to_global: Ego-to-global pose for this frame.
    :return: The box-centre pose in the global frame.
    """
    box_in_ego = np.eye(4, dtype=np.float64)
    box_in_ego[:3, :3] = euler_xyz_to_rotation_matrix(annotation["roll"], annotation["pitch"], annotation["yaw"])
    box_in_ego[:3, 3] = [annotation["x"], annotation["y"], annotation["z"]]
    return rel_to_abs_se3(origin=ego_to_global, pose_se3=PoseSE3.from_transformation_matrix(box_in_ego))
