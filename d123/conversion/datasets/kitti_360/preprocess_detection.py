"""
This script precomputes static detection records for KITTI-360:
  - Stage 1: radius filtering using ego positions (from poses.txt).
  - Stage 2: LiDAR visibility check to fill per-frame point counts.
It writes a pickle containing, for each static object, all feasible frames and
their point counts to avoid recomputation in later pipelines.
We have precomputed and saved the pickle for all training logs, you can either
download them or run this script to generate
"""

from __future__ import annotations
import os
import pickle
import logging
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import concurrent.futures

import numpy as np
import numpy.typing as npt
import xml.etree.ElementTree as ET

KITTI360_DATA_ROOT = Path(os.environ["KITTI360_DATA_ROOT"])
DIR_3D_RAW = "data_3d_raw"
DIR_3D_BBOX = "data_3d_bboxes"
DIR_POSES = "data_poses"

PATH_3D_RAW_ROOT = KITTI360_DATA_ROOT / DIR_3D_RAW
PATH_3D_BBOX_ROOT = KITTI360_DATA_ROOT / DIR_3D_BBOX
PATH_POSES_ROOT = KITTI360_DATA_ROOT / DIR_POSES

from d123.conversion.datasets.kitti_360.kitti_360_helper import KITTI360Bbox3D, KITTI3602NUPLAN_IMU_CALIBRATION, get_lidar_extrinsic
from d123.conversion.datasets.kitti_360.labels import KITTI360_DETECTION_NAME_DICT, kittiId2label, BBOX_LABLES_TO_DETECTION_NAME_DICT

def _bbox_xml_path(log_name: str) -> Path:
    if log_name == "2013_05_28_drive_0004_sync":
        return PATH_3D_BBOX_ROOT / "train_full" / f"{log_name}.xml"
    return PATH_3D_BBOX_ROOT / "train" / f"{log_name}.xml"

def _lidar_frame_path(log_name: str, frame_idx: int) -> Path:
    return PATH_3D_RAW_ROOT / log_name / "velodyne_points" / "data" / f"{frame_idx:010d}.bin"

def _load_lidar_xyz(filepath: Path) -> np.ndarray:
    """Load one LiDAR frame and return Nx3 xyz."""
    arr = np.fromfile(filepath, dtype=np.float32)
    return arr.reshape(-1, 4)[:, :3]

def _collect_static_objects(log_name: str) -> List[KITTI360Bbox3D]:
    """Parse XML and collect static objects with valid class names."""
    xml_path = _bbox_xml_path(log_name)
    if not xml_path.exists():
        raise FileNotFoundError(f"BBox 3D file not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    static_objs: List[KITTI360Bbox3D] = []

    for child in root:
        if child.find('semanticId') is not None:
            semanticIdKITTI = int(child.find('semanticId').text)
            name = kittiId2label[semanticIdKITTI].name
        else:
            lable = child.find('label').text
            name = BBOX_LABLES_TO_DETECTION_NAME_DICT.get(lable, 'unknown')
        timestamp = int(child.find('timestamp').text)  # -1 for static objects
        if child.find("transform") is None or name not in KITTI360_DETECTION_NAME_DICT or timestamp != -1:
            continue
        obj = KITTI360Bbox3D()
        obj.parseBbox(child)
        static_objs.append(obj)
    return static_objs

def _collect_ego_states(log_name: str) -> Tuple[npt.NDArray[np.float64], list[int]]:
    """Load ego states from poses.txt."""

    pose_file = PATH_POSES_ROOT / log_name / "poses.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    
    poses = np.loadtxt(pose_file)
    poses_time = poses[:, 0].astype(np.int32)
    valid_timestamp: List[int] = list(poses_time)
    
    ego_states = []
    for time_idx in range(len(valid_timestamp)):
        pos = time_idx
        state_item = np.eye(4)
        r00, r01, r02 = poses[pos, 1:4]
        r10, r11, r12 = poses[pos, 5:8]
        r20, r21, r22 = poses[pos, 9:12]
        R_mat = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]], dtype=np.float64)
        R_mat_cali = R_mat @ KITTI3602NUPLAN_IMU_CALIBRATION[:3,:3]
        ego_state_xyz = np.array([
            poses[pos, 4],
            poses[pos, 8],
            poses[pos, 12],
        ])

        state_item[:3, :3] = R_mat_cali
        state_item[:3, 3] = ego_state_xyz
        ego_states.append(state_item)

    # [N,4,4]
    return np.array(ego_states), valid_timestamp


def process_detection(
    log_name: str,
    radius_m: float = 60.0,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Precompute detections filtering
    for static objects:
      1) filter by ego-centered radius over all frames
      2) filter by LiDAR point cloud visibility
    Save per-frame detections to a pickle to avoid recomputation.
    """

    lidar_dir = PATH_3D_RAW_ROOT / log_name / "velodyne_points" / "data"
    if not lidar_dir.exists():
        raise FileNotFoundError(f"LiDAR data folder not found: {lidar_dir}")
    ts_len = len(list(lidar_dir.glob("*.bin")))
    logging.info(f"[preprocess] {log_name}: found {ts_len} lidar frames")

    # 1) Parse objects from XML
    static_objs: List[KITTI360Bbox3D] = _collect_static_objects(log_name)
    logging.info(f"[preprocess] {log_name}: static objects = {len(static_objs)}")

    # 2) Filter static objs by ego-centered radius
    ego_states, valid_timestamp = _collect_ego_states(log_name)
    logging.info(f"[preprocess] {log_name}: ego states = {len(ego_states)}")
    for obj in static_objs:
        obj.filter_by_radius(ego_states[:, :3, 3], valid_timestamp, radius_m)

    # 3) Filter static objs by LiDAR point cloud visibility
    lidar_extrinsic = get_lidar_extrinsic()

    def process_one_frame(time_idx: int) -> None:
        valid_time_idx = valid_timestamp[time_idx]
        logging.info(f"[preprocess] {log_name}: t={valid_time_idx}")
        lidar_path = _lidar_frame_path(log_name, valid_time_idx)
        if not lidar_path.exists():
            logging.warning(f"[preprocess] {log_name}: LiDAR frame not found: {lidar_path}")
            return
        
        lidar_xyz = _load_lidar_xyz(lidar_path)

        # lidar to pose
        lidar_h = np.concatenate((lidar_xyz, np.ones((lidar_xyz.shape[0], 1), dtype=lidar_xyz.dtype)), axis=1)
        lidar_in_imu = lidar_h @ lidar_extrinsic.T
        lidar_in_imu = lidar_in_imu[:,:3]

        # pose to world
        lidar_in_world = lidar_in_imu @ ego_states[time_idx][:3,:3].T + ego_states[time_idx][:3,3]

        for obj in static_objs:
            if not any(record["timestamp"] == valid_time_idx for record in obj.valid_frames["records"]):
                continue
            visible, points_in_box = obj.box_visible_in_point_cloud(lidar_in_world)
            if not visible:
                obj.valid_frames["records"] = [record for record in obj.valid_frames["records"] if record["timestamp"] != valid_time_idx]
            else:
                for record in obj.valid_frames["records"]:
                    if record["timestamp"] == valid_time_idx:
                        record["points_in_box"] = points_in_box
                        break

    max_workers = os.cpu_count() * 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_one_frame, range(len(valid_timestamp))))

    # 4) Save pickle
    static_records: List[Dict[str, Any]] = []
    for obj in static_objs:
        static_records.append(obj.valid_frames)

    if output_dir is None:
        output_dir = PATH_3D_BBOX_ROOT / "preprocess"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{log_name}_detection_preprocessed.pkl"

    payload = {
        "log_name": log_name,
        "static": static_records,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    logging.info(f"[preprocess] saved: {out_path}")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Precompute KITTI-360 detections filters")
    parser.add_argument("--log_name", default="2013_05_28_drive_0000_sync")
    parser.add_argument("--radius", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default="detection_preprocess", help="output directory for pkl")
    args = parser.parse_args()

    process_detection(
        log_name=args.log_name,
        radius_m=args.radius,
        output_dir=args.out,
    )
