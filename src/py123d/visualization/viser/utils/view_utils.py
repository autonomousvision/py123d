from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.sensors.base_camera import Camera
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import EulerAngles, PoseSE3Index, Vector3D
from py123d.geometry.pose import PoseSE3
from py123d.geometry.rotation import Quaternion
from py123d.geometry.transform.transform_se3 import abs_to_rel_se3_array, translate_se3_along_body_frame
from py123d.parser.utils.sensor_utils.camera_conventions import convert_camera_convention


def decompose_camera_pose(
    camera: Camera, scene_center_pose: PoseSE3
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decompose a camera's global pose into position and quaternion relative to the scene center."""
    global_camera_se3 = camera.camera_to_global_se3.array
    abs_camera_pose = abs_to_rel_se3_array(origin=scene_center_pose, pose_se3_array=global_camera_se3)
    return abs_camera_pose[PoseSE3Index.XYZ], abs_camera_pose[PoseSE3Index.QUATERNION]


def get_scene_center_pose(scene_center_array: npt.NDArray[np.float64]) -> PoseSE3:
    """Create a PoseSE3 at the scene center with identity rotation."""
    return PoseSE3.from_R_t(rotation=Quaternion.identity(), translation=scene_center_array)


def get_ego_3rd_person_view_position(
    scene: SceneAPI,
    iteration: int,
    initial_ego_state: EgoStateSE3,
) -> PoseSE3:
    """Position camera 15m behind and 15m above ego vehicle with 30 degree pitch."""
    scene_center_array = initial_ego_state.center_se3.point_3d.array
    ego_center = scene.get_ego_state_se3_at_iteration(iteration).center_se3.array.copy()
    ego_center[PoseSE3Index.XYZ] -= scene_center_array
    ego_pose_se3 = PoseSE3.from_array(ego_center)

    planar_euler_angles = EulerAngles(0.0, 0.0, _get_planar_heading(scene, iteration))
    ego_pose_se3._array[PoseSE3Index.QUATERNION] = planar_euler_angles.quaternion.array

    ego_pose_se3 = translate_se3_along_body_frame(ego_pose_se3, Vector3D(-10.0, 0.0, 9.0))
    ego_pose_se3 = _pitch_se3_by_degrees(ego_pose_se3, 25.0)

    return convert_camera_convention(
        ego_pose_se3,
        from_convention="pXpZmY",
        to_convention="pZmYpX",
    )


def get_ego_bev_view_position(
    scene: SceneAPI,
    iteration: int,
    initial_ego_state: EgoStateSE3,
) -> PoseSE3:
    """Position camera 50m directly above ego vehicle looking straight down."""
    scene_center_array = initial_ego_state.center_se3.point_3d.array
    ego_center = scene.get_ego_state_se3_at_iteration(iteration).center_se3.array
    ego_center[PoseSE3Index.XYZ] -= scene_center_array
    ego_center_planar = PoseSE3.from_array(ego_center)

    planar_euler_angles = EulerAngles(0.0, 0.0, ego_center_planar.euler_angles.yaw)
    quaternion = planar_euler_angles.quaternion
    ego_center_planar._array[PoseSE3Index.QUATERNION] = quaternion.array

    ego_center_planar = translate_se3_along_body_frame(ego_center_planar, Vector3D(0.0, 0.0, 50))
    ego_center_planar = _pitch_se3_by_degrees(ego_center_planar, 90.0)

    return convert_camera_convention(
        ego_center_planar,
        from_convention="pXpZmY",
        to_convention="pZmYpX",
    )


def _pitch_se3_by_degrees(pose_se3: PoseSE3, degrees: float) -> PoseSE3:
    quaternion = EulerAngles(0.0, np.deg2rad(degrees), pose_se3.yaw).quaternion

    return PoseSE3(
        x=pose_se3.x,
        y=pose_se3.y,
        z=pose_se3.z,
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )


# The heading at a frame is the direction of the path chord spanning +/- _PATH_WINDOW_M
# of arc length around it: slow motion uses a long time window, driving a short one, so
# the heading is defined by path geometry instead of frame-to-frame localization noise.
# Only frames where even that chord is shorter than _MIN_PATH_MOTION_M (static scene,
# reversal cusp) fall back to the ego yaw.
_PATH_WINDOW_M = 2.0
_MIN_PATH_MOTION_M = 0.5
_HEADING_SMOOTH_HALF_WINDOW = 5  # frames on each side in the circular moving average

_path_heading_cache: Dict[str, npt.NDArray[np.float64]] = {}


def _compute_path_headings(scene: SceneAPI) -> npt.NDArray[np.float64]:
    num_iterations = scene.number_of_iterations
    xy = np.empty((num_iterations, 2), dtype=np.float64)
    ego_yaw = np.empty(num_iterations, dtype=np.float64)
    for i in range(num_iterations):
        state = scene.get_ego_state_se3_at_iteration(i)
        assert state is not None, "Ego state must be available at every iteration."
        xy[i] = state.center_se3.array[PoseSE3Index.XY]
        ego_yaw[i] = state.center_se3.yaw

    if num_iterations < 2:
        return ego_yaw

    arc_length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])

    headings = np.empty(num_iterations, dtype=np.float64)
    for i in range(num_iterations):
        forward = min(int(np.searchsorted(arc_length, arc_length[i] + _PATH_WINDOW_M, side="left")), num_iterations - 1)
        backward = max(int(np.searchsorted(arc_length, arc_length[i] - _PATH_WINDOW_M, side="right")) - 1, 0)
        chord_xy = xy[forward] - xy[backward]
        if np.linalg.norm(chord_xy) < _MIN_PATH_MOTION_M:
            headings[i] = ego_yaw[i]
        else:
            headings[i] = np.arctan2(chord_xy[1], chord_xy[0])

    # Circular moving average on unit vectors; plain averaging would break at the +/-pi wrap.
    directions = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    padded = np.pad(directions, ((_HEADING_SMOOTH_HALF_WINDOW, _HEADING_SMOOTH_HALF_WINDOW), (0, 0)), mode="edge")
    kernel = np.ones(2 * _HEADING_SMOOTH_HALF_WINDOW + 1) / (2 * _HEADING_SMOOTH_HALF_WINDOW + 1)
    smoothed_cos = np.convolve(padded[:, 0], kernel, mode="valid")
    smoothed_sin = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.arctan2(smoothed_sin, smoothed_cos)


def _get_planar_heading(scene: SceneAPI, iteration: int) -> float:
    headings = _path_heading_cache.get(scene.scene_uuid)
    if headings is None:
        headings = _compute_path_headings(scene)
        _path_heading_cache[scene.scene_uuid] = headings
    return float(headings[iteration])
