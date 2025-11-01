"""Helper functions for NUREC dataset parsing."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def get_available_camera(scene_path: Path) -> Optional[Tuple[str, Path]]:
    """
    Find the available camera video file in the scene.

    NUREC provides multiple camera descriptions in metadata but only one MP4 file.
    This function finds the actual camera video that exists.

    Args:
        scene_path: Path to the scene directory

    Returns:
        Tuple of (camera_name, video_path) if found, None otherwise
    """
    # Look for any MP4 file that matches camera naming pattern
    mp4_files = list(scene_path.glob("camera_*.mp4"))

    if len(mp4_files) == 0:
        return None

    if len(mp4_files) > 1:
        # If multiple cameras exist, prefer front wide
        for mp4_file in mp4_files:
            if "front_wide" in mp4_file.stem:
                return mp4_file.stem, mp4_file
        # Otherwise return the first one
        return mp4_files[0].stem, mp4_files[0]

    return mp4_files[0].stem, mp4_files[0]


def transformation_matrix_to_pose_7d(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert a 4x4 transformation matrix to 7D pose [x, y, z, qw, qx, qy, qz].

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        7D pose array [x, y, z, qw, qx, qy, qz]
    """
    from scipy.spatial.transform import Rotation

    # Extract translation
    translation = matrix[:3, 3]

    # Extract rotation matrix and convert to quaternion
    rotation_matrix = matrix[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]

    # Convert to [x, y, z, qw, qx, qy, qz] format
    pose_7d = np.array(
        [
            translation[0],
            translation[1],
            translation[2],
            quaternion[3],  # qw
            quaternion[0],  # qx
            quaternion[1],  # qy
            quaternion[2],  # qz
        ],
        dtype=np.float64,
    )

    return pose_7d


def find_nearest_timestamp_index(
    timestamps: npt.NDArray[np.int64],
    target_timestamp: int,
    max_time_diff_us: int = 50000,  # 50ms default
) -> Optional[int]:
    """
    Find the index of the nearest timestamp within a maximum time difference.

    Args:
        timestamps: Array of timestamps in microseconds
        target_timestamp: Target timestamp in microseconds
        max_time_diff_us: Maximum allowed time difference in microseconds

    Returns:
        Index of nearest timestamp if within max_time_diff, None otherwise
    """
    if len(timestamps) == 0:
        return None

    # Find the index of the closest timestamp
    time_diffs = np.abs(timestamps - target_timestamp)
    min_idx = int(np.argmin(time_diffs))

    # Check if within threshold
    if time_diffs[min_idx] <= max_time_diff_us:
        return min_idx

    return None


def interpolate_pose(
    timestamp: int,
    timestamps: npt.NDArray[np.int64],
    poses: List[npt.NDArray[np.float64]],
) -> Optional[npt.NDArray[np.float64]]:
    """
    Interpolate a pose at a given timestamp using linear interpolation.

    Args:
        timestamp: Target timestamp in microseconds
        timestamps: Array of pose timestamps in microseconds
        poses: List of 4x4 transformation matrices

    Returns:
        Interpolated 4x4 transformation matrix, or None if timestamp out of range
    """
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        return None

    # Find surrounding indices
    idx = np.searchsorted(timestamps, timestamp)

    if idx == 0:
        return poses[0]
    if idx >= len(timestamps):
        return poses[-1]

    # Linear interpolation factor
    t0 = timestamps[idx - 1]
    t1 = timestamps[idx]
    alpha = (timestamp - t0) / (t1 - t0)

    # Interpolate translation
    trans0 = poses[idx - 1][:3, 3]
    trans1 = poses[idx][:3, 3]
    trans_interp = trans0 + alpha * (trans1 - trans0)

    # SLERP for rotation
    from scipy.spatial.transform import Rotation, Slerp

    rot0 = Rotation.from_matrix(poses[idx - 1][:3, :3])
    rot1 = Rotation.from_matrix(poses[idx][:3, :3])

    slerp = Slerp([0, 1], Rotation.concatenate([rot0, rot1]))
    rot_interp = slerp([alpha])[0]

    # Construct interpolated matrix
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rot_interp.as_matrix()
    result[:3, 3] = trans_interp

    return result


def get_scene_origin_from_pose_record(pose_record: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract the scene origin (alignment origin) from pose record.

    Args:
        pose_record: Loaded pose_record.json data

    Returns:
        Dictionary with latitude, longitude, altitude
    """
    return pose_record.get(
        "alignment_origin",
        {
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude": 0.0,
        },
    )


def extract_video_frame(video_path: Path, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
    """
    Extract a specific frame from a video file.

    Args:
        video_path: Path to the MP4 video file
        frame_index: Zero-based frame index to extract

    Returns:
        Frame as BGR image array (H, W, 3), or None if extraction fails
    """
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))

    try:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read frame
        ret, frame = cap.read()

        if not ret or frame is None:
            return None

        return frame

    finally:
        cap.release()


def get_video_info(video_path: Path) -> Optional[Dict[str, Any]]:
    """
    Get video metadata.

    Args:
        video_path: Path to the MP4 video file

    Returns:
        Dictionary with video info (frame_count, fps, width, height), or None if fails
    """
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
        }

    finally:
        cap.release()


def find_nearest_camera_frame_index(
    camera_timestamps: List[List[int]],
    target_timestamp_us: int,
    max_time_diff_us: int = 50000,  # 50ms default
) -> Optional[int]:
    """
    Find the nearest camera frame index for a target timestamp.

    NUREC camera timestamps are stored as pairs [start, end] (for what can be
    assumed is rolling shutter). Use the end timestamp for now (when capture is
    complete).

    TODO : Maybe use nearest point search between start and end timestamps.

    Args:
        camera_timestamps: List of [start_us, end_us] timestamp pairs
        target_timestamp_us: Target timestamp in microseconds
        max_time_diff_us: Maximum allowed time difference in microseconds

    Returns:
        Frame index if found within threshold, None otherwise
    """
    if len(camera_timestamps) == 0:
        return None

    # Extract end timestamps (second value in each pair)
    end_timestamps = np.array([ts[1] for ts in camera_timestamps], dtype=np.int64)

    # Find nearest
    time_diffs = np.abs(end_timestamps - target_timestamp_us)
    min_idx = int(np.argmin(time_diffs))

    # Check if within threshold
    if time_diffs[min_idx] <= max_time_diff_us:
        return min_idx

    return None
