import gc
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import datetime
import hashlib
import pyarrow as pa
from PIL import Image
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

from d123.common.datatypes.sensor.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.common.datatypes.time.time_point import TimePoint
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE3Index
from d123.common.datatypes.vehicle_state.vehicle_parameters import get_kitti360_station_wagon_parameters
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3Index
from d123.common.geometry.vector import Vector3DIndex
from d123.dataset.arrow.helper import open_arrow_table, write_arrow_table
from d123.dataset.dataset_specific.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.dataset.logs.log_metadata import LogMetadata

KITTI360_DT: Final[float] = 0.1
SORT_BY_TIMESTAMP: Final[bool] = True

KITTI360_DATA_ROOT = Path(os.environ["KITTI360_DATA_ROOT"])

#TODO  carera mismatch
KITTI360_CAMERA_TYPES: Final[Dict[CameraType, str]] = {
    CameraType.CAM_L0: "image_00",  
    CameraType.CAM_R0: "image_01",   
    # TODO fisheye camera
    # CameraType.CAM_L1: "image_02", 
    # CameraType.CAM_R1: "image_03", 
}

DIR_2D_RAW = "data_2d_raw"
DIR_2D_SMT = "data_2d_semantics"
DIR_3D_RAW = "data_3d_raw"
DIR_3D_SMT = "data_3d_semantics"
DIR_3D_BBOX = "data_3d_bboxes"
DIR_POSES = "data_poses"
DIR_CALIB = "calibration"

#TODO PATH_2D_RAW_ROOT
PATH_2D_RAW_ROOT: Path = KITTI360_DATA_ROOT 
PATH_2D_SMT_ROOT: Path = KITTI360_DATA_ROOT / DIR_2D_SMT
PATH_3D_RAW_ROOT: Path = KITTI360_DATA_ROOT / DIR_3D_RAW
PATH_3D_SMT_ROOT: Path = KITTI360_DATA_ROOT / DIR_3D_SMT
PATH_3D_BBOX_ROOT: Path = KITTI360_DATA_ROOT / DIR_3D_BBOX
PATH_POSES_ROOT: Path = KITTI360_DATA_ROOT / DIR_POSES
PATH_CALIB_ROOT: Path = KITTI360_DATA_ROOT / DIR_CALIB

KITTI360_REQUIRED_MODALITY_ROOTS: Dict[str, Path] = {
    DIR_2D_RAW: PATH_2D_RAW_ROOT,
    # DIR_2D_SMT: PATH_2D_SMT_ROOT,
    # DIR_3D_RAW: PATH_3D_RAW_ROOT,
    # DIR_3D_SMT: PATH_3D_SMT_ROOT,
    # DIR_3D_BBOX: PATH_3D_BBOX_ROOT,
    # DIR_POSES: PATH_POSES_ROOT,
}


def create_token(input_data: str) -> str:
    # TODO: Refactor this function.
    # TODO: Add a general function to create tokens from arbitrary data.
    if isinstance(input_data, str):
        input_data = input_data.encode("utf-8")

    hash_obj = hashlib.sha256(input_data)
    return hash_obj.hexdigest()[:16]

def _load_calibration() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    读取 KITTI-360 全局标定文件，返回:
    - intrinsics[image_02] = 3x3
    - c2e[image_02] = 4x4（camera->ego/body），这里将 cam_to_pose 视为 camera->vehicle（简化）
    """
    calib_dir = KITTI360_DATA_ROOT / DIR_CALIB
    intrinsics: Dict[str, np.ndarray] = {}
    c2e: Dict[str, np.ndarray] = {}

    # 内参：perspective.txt 中的 P_rect_0{0..3}
    persp = calib_dir / "perspective.txt"
    if persp.exists():
        with open(persp, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            if ln.startswith("P_rect_02"):
                intrinsics["image_02"] = _read_projection_matrix(ln)
            elif ln.startswith("P_rect_03"):
                intrinsics["image_03"] = _read_projection_matrix(ln)

    # 外参：cam_to_pose.txt 中 Tr_cam02（相机到车体/pose）
    c2p = calib_dir / "cam_to_pose.txt"
    if c2p.exists():
        with open(c2p, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            if ln.startswith("Tr_cam02"):
                vals = [float(x) for x in ln.split(":")[1].strip().split()]
                T = np.array(vals, dtype=np.float64).reshape(4, 4)
                c2e["image_02"] = T
            elif ln.startswith("Tr_cam03"):
                vals = [float(x) for x in ln.split(":")[1].strip().split()]
                T = np.array(vals, dtype=np.float64).reshape(4, 4)
                c2e["image_03"] = T

    return intrinsics, c2e

class Kitti360DataConverter(RawDataConverter):
    def __init__(
        self,
        splits: List[str],
        log_path: Union[Path, str],
        data_converter_config: DataConverterConfig,
    ) -> None:
        super().__init__(data_converter_config)
        for split in splits:
            assert (
                split in self.get_available_splits()
            ), f"Split {split} is not available. Available splits: {self.available_splits}"

        self._splits: List[str] = splits
        self._log_path: Path = Path(log_path)
        self._log_paths_per_split: Dict[str, List[Path]] = self._collect_log_paths()

    def _collect_log_paths(self) -> Dict[str, List[Path]]:
        """
            Collect candidate sequence folders under data_2d_raw that end with '_sync',
            and keep only those sequences that are present in ALL required modality roots
            (e.g., data_2d_semantics, data_3d_raw, etc.).
        """
        missing_roots = [str(p) for p in KITTI360_REQUIRED_MODALITY_ROOTS.values() if not p.exists()]
        if missing_roots:
            raise FileNotFoundError(f"KITTI-360 required roots missing: {missing_roots}")
    
        # Enumerate candidate sequences from data_2d_raw
        candidates = sorted(p for p in PATH_2D_RAW_ROOT.iterdir() if p.is_dir() and p.name.endswith("_sync"))

        valid_seqs: List[Path] = []
        for seq_dir in candidates:
            seq_name = seq_dir.name
            missing_modalities = [
                modality_name
                for modality_name, root in KITTI360_REQUIRED_MODALITY_ROOTS.items()
                if not (root / seq_name).exists()
            ]
            if not missing_modalities:
                valid_seqs.append(seq_dir) #KITTI360_DATA_ROOT / DIR_2D_RAW /seq_name
            #TODO warnings
            # else:
            #     warnings.warn(
            #         f"Sequence '{seq_name}' skipped: missing modalities {missing_modalities}. "
            #         f"Root: {KITTI360_DATA_ROOT}"
            #     )
        return {"kitti360": valid_seqs}
    
    def get_available_splits(self) -> List[str]:
        """Returns a list of available raw data types."""
        return ["kitti360"]

    def convert_maps(self, worker: WorkerPool) -> None:
        print("KITTI-360 does not provide standard maps. Skipping map conversion.")
        return None

    def convert_logs(self, worker: WorkerPool) -> None:
        log_args = [
            {
                "log_path": log_path,
                "split": split,
            }
            for split, log_paths in self._log_paths_per_split.items()
            for log_path in log_paths
        ]

        worker_map(
            worker,
            partial(
                convert_kitti360_log_to_arrow,
                data_converter_config=self.data_converter_config,
            ),
            log_args,
        )

def convert_kitti360_log_to_arrow(
    args: List[Dict[str, Union[List[str], List[Path]]]], data_converter_config: DataConverterConfig
) -> List[Any]:
    
    for log_info in args:
        log_path: Path = log_info["log_path"]
        split: str = log_info["split"]
        log_name = log_path.stem

        if not log_path.exists():
            raise FileNotFoundError(f"Log path {log_path} does not exist.")
        log_file_path = data_converter_config.output_path / split / f"{log_name}.arrow"
       
        if data_converter_config.force_log_conversion or not log_file_path.exists():
            log_file_path.unlink(missing_ok=True)
            if not log_file_path.parent.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            schema_column_list = [
                ("token", pa.string()),
                ("timestamp", pa.int64()),
                ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                ("detections_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                ("detections_token", pa.list_(pa.string())),
                ("detections_type", pa.list_(pa.int16())),
                ("ego_states", pa.list_(pa.float64(), len(EgoStateSE3Index))),
                ("traffic_light_ids", pa.list_(pa.int64())),
                ("traffic_light_types", pa.list_(pa.int16())),
                ("scenario_tag", pa.list_(pa.string())),
                ("route_lane_group_ids", pa.list_(pa.int64())),
            ]
            if data_converter_config.lidar_store_option is not None:
                if data_converter_config.lidar_store_option == "path":
                    schema_column_list.append(("lidar", pa.string()))
                elif data_converter_config.lidar_store_option == "binary":
                    raise NotImplementedError("Binary lidar storage is not implemented.")

            # TODO: Adjust how cameras are added
            if data_converter_config.camera_store_option is not None:
                for cam_type in KITTI360_CAMERA_TYPES.keys():
                    if data_converter_config.camera_store_option == "path":
                        schema_column_list.append((cam_type.serialize(), pa.string()))
                        schema_column_list.append((f"{cam_type.serialize()}_extrinsic", pa.list_(pa.float64(), 16)))
                    elif data_converter_config.camera_store_option == "binary":
                        raise NotImplementedError("Binary camera storage is not implemented.")

            recording_schema = pa.schema(schema_column_list)
            #TODO location
            metadata = LogMetadata(
                dataset="kitti360",
                log_name=log_name,
                location="None",
                timestep_seconds=KITTI360_DT,
                map_has_z=False,
            )

            #TODO vehicle parameters
            vehicle_parameters = get_kitti360_station_wagon_parameters()
            camera_metadata = get_kitti360_camera_metadata()
            recording_schema = recording_schema.with_metadata(
                {
                    "log_metadata": json.dumps(asdict(metadata)),
                    "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                    "camera_metadata": camera_metadata_dict_to_json(camera_metadata),
                }
            )

            _write_recording_table(log_name, recording_schema, log_file_path, data_converter_config)

        gc.collect()
    return []


def get_kitti360_camera_metadata() -> Dict[str, CameraMetadata]:
    
    persp = PATH_CALIB_ROOT / "perspective.txt"

    assert persp.exists()
    result = {"image_00": {}, "image_01": {}}

    with open(persp, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            key, value = ln.split(" ", 1)
            cam_id = key.split("_")[-1][:2]
            if key.startswith("P_rect_"):
                result[f"image_{cam_id}"]["intrinsic"] = _read_projection_matrix(ln)
            elif key.startswith("S_rect_"):
                result[f"image_{cam_id}"]["wh"] = [int(round(float(x))) for x in value.split()]
            elif key.startswith("D_"):
                result[f"image_{cam_id}"]["distortion"] = [float(x) for x in value.split()]
    
    log_cam_infos: Dict[str, CameraMetadata] = {}
    for cam_type, cam_name in KITTI360_CAMERA_TYPES.items():
        log_cam_infos[cam_type.serialize()] = CameraMetadata(
            camera_type=cam_type,
            width=result[cam_name]["wh"][0],
            height=result[cam_name]["wh"][1],
            intrinsic=np.array(result[cam_name]["intrinsic"]),
            distortion=np.array(result[cam_name]["distortion"]),
        )
    return log_cam_infos

def _read_projection_matrix(p_line: str) -> np.ndarray:
    parts = p_line.split(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad projection line: {p_line}")
    vals = [float(x) for x in parts[1].strip().split()]
    P = np.array(vals, dtype=np.float64).reshape(3, 4)
    K = P[:, :3]
    return K

def _write_recording_table(
    log_name: str,
    recording_schema: pa.Schema,
    log_file_path: Path,
    data_converter_config: DataConverterConfig
) -> None:
    
    ts_list = _read_timestamps(log_name)

    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:
            for i, tp in enumerate(ts_list):
                row_data = {
                    "token": [create_token(f"{log_name}_{i}")],
                    "timestamp": [tp.time_us],
                    "detections_state": [],
                    "detections_velocity": [],
                    "detections_token": [],
                    "detections_type": [],
                    "ego_states": [],
                    "traffic_light_ids": [],
                    "traffic_light_types": [],
                    "scenario_tag": [],
                    "route_lane_group_ids": [],
                }

                if data_converter_config.lidar_store_option is not None:
                    row_data["lidar"] = []
                    # row_data["lidar"] = [_extract_lidar(log_name, data_converter_config)]

                if data_converter_config.camera_store_option is not None:
                    # camera_data_dict = _extract_camera(log_db, lidar_pc, source_log_path, data_converter_config)
                    camera_data_dict = {}
                    for camera_type, camera_data in camera_data_dict.items():
                        if camera_data is not None:
                            row_data[camera_type.serialize()] = [camera_data[0]]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [camera_data[1]]
                        else:
                            row_data[camera_type.serialize()] = [None]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [None]

                batch = pa.record_batch(row_data, schema=recording_schema)
                writer.write_batch(batch)

    if SORT_BY_TIMESTAMP:
        recording_table = open_arrow_table(log_file_path)
        recording_table = recording_table.sort_by([("timestamp", "ascending")])
        write_arrow_table(recording_table, log_file_path)

#TODO default timestamps
# If timestamps are not provided, we can generate them based on the KITTI-360 DT
def _read_timestamps(log_name: str) -> Optional[List[TimePoint]]:
    
    ts_file = PATH_2D_RAW_ROOT / log_name / "image_01" / "timestamps.txt"
    if ts_file.exists():
        tps: List[TimePoint] = []
        with open(ts_file, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                dt_str, ns_str = s.split('.')
                dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
                unix_epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
                
                total_seconds = (dt_obj - unix_epoch).total_seconds()
                
                ns_value = int(ns_str)
                us_from_ns = ns_value // 1000

                total_us = int(total_seconds * 1_000_000) + us_from_ns
                
                tps.append(TimePoint.from_us(total_us))
        return tps
    return None

#TODO lidar extraction
def _extract_lidar(log_name: str, data_converter_config: DataConverterConfig) -> Optional[str]:
    lidar: Optional[str] = None
    lidar_full_path = DIR_3D_SMT / "train" / log_name / "0000000002_0000000385.ply"
    if lidar_full_path.exists():
        if data_converter_config.lidar_store_option == "path":
            lidar = f"{log_name}/lidar/{sample_name}.npy"
        elif data_converter_config.lidar_store_option == "binary":
            raise NotImplementedError("Binary lidar storage is not implemented.")
    else:
        raise FileNotFoundError(f"LiDAR file not found: {lidar_full_path}")
    return lidar

def _extract_camera():
    pass



#                     for idx in range(n_frames):
#                         token = f"{seq_name}_{idx:06d}"
#                         t_us = ts_list[idx].time_us

#                         row = {
#                             "token": [token],
#                             "timestamp": [t_us],
#                             # 以下先填空/占位，方便后续替换为真实标注
#                             "detections_state": [[]],
#                             "detections_velocity": [[]],
#                             "detections_token": [[]],
#                             "detections_type": [[]],
#                             "ego_states": [([0.0] * len(EgoStateSE3Index))],  # 占位
#                             "traffic_light_ids": [[]],
#                             "traffic_light_types": [[]],
#                             "scenario_tag": [["unknown"]],
#                             "route_lane_group_ids": [[]],
#                         }

#                         # lidar 路径（若存在）
#                         if data_converter_config.lidar_store_option is not None:
#                             # velodyne bin：KITTI-360/data_3d_raw/<seq>/velodyne_points/data/0000000000.bin
#                             velodyne_dir = (
#                                 KITTI360_DATA_ROOT / DIR_3D / seq_name / "velodyne_points" / "data"
#                             )
#                             # 文件名位数可能为 10 位，这里做两种尝试
#                             bin_path = None
#                             for fmt in [f"{idx:010d}.bin", f"{idx:06d}.bin", f"{idx:08d}.bin"]:
#                                 cand = velodyne_dir / fmt
#                                 if cand.exists():
#                                     bin_path = cand
#                                     break
#                             row["lidar"] = [str(bin_path.relative_to(KITTI360_DATA_ROOT)) if bin_path else None]

#                         # 相机路径与外参
#                         if data_converter_config.camera_store_option is not None:
#                             for cam_type, cam_dir_name in KITTI360_CAMERA_TYPES.items():
#                                 img_dir = seq_dir_2d / cam_dir_name / "data"
#                                 # 文件名位数尝试
#                                 img_path = None
#                                 for ext in (".png", ".jpg", ".jpeg"):
#                                     for fmt in [f"{idx:010d}{ext}", f"{idx:06d}{ext}", f"{idx:08d}{ext}"]:
#                                         cand = img_dir / fmt
#                                         if cand.exists():
#                                             img_path = cand
#                                             break
#                                     if img_path:
#                                         break
#                                 if img_path is not None:
#                                     rel = str(img_path.relative_to(KITTI360_DATA_ROOT))
#                                     row[cam_type.serialize()] = [rel]
#                                     # 外参：固定 cam->ego（全局标定），逐帧不变（如需 rolling/姿态，可在此替换）
#                                     T = c2e.get(KITTI360_CAMERA_TYPES[cam_type], np.eye(4, dtype=np.float64))
#                                     row[f"{cam_type.serialize()}_extrinsic"] = [T.astype(np.float64).reshape(-1).tolist()]
#                                 else:
#                                     row[cam_type.serialize()] = [None]
#                                     row[f"{cam_type.serialize()}_extrinsic"] = [None]

#                         batch = pa.record_batch(row, schema=recording_schema)
#                         writer.write_batch(batch)
#                         del batch, row