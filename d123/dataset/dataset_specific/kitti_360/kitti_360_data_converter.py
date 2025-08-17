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
import xml.etree.ElementTree as ET
import pyarrow as pa
from PIL import Image
import logging

from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

from d123.common.datatypes.detection.detection_types import DetectionType
from d123.common.datatypes.sensor.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar import LiDARMetadata, LiDARType, lidar_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar_index import Kitti360LidarIndex
from d123.common.datatypes.time.time_point import TimePoint
from d123.common.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3, EgoStateSE3Index
from d123.common.datatypes.vehicle_state.vehicle_parameters import get_kitti360_station_wagon_parameters,rear_axle_se3_to_center_se3
from d123.common.geometry.base import StateSE3
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3Index
from d123.common.geometry.vector import Vector3D, Vector3DIndex
from d123.dataset.arrow.helper import open_arrow_table, write_arrow_table
from d123.dataset.dataset_specific.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.dataset.logs.log_metadata import LogMetadata
from d123.dataset.dataset_specific.kitti_360.kitti_360_helper import KITTI360Bbox3D

KITTI360_DT: Final[float] = 0.1
SORT_BY_TIMESTAMP: Final[bool] = True

KITTI360_DATA_ROOT = Path(os.environ["KITTI360_DATA_ROOT"])

#TODO  carera mismatch
KITTI360_CAMERA_TYPES = {
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
    DIR_3D_RAW: PATH_3D_RAW_ROOT,
    DIR_POSES: PATH_POSES_ROOT,
    DIR_3D_BBOX: PATH_3D_BBOX_ROOT / "train",
}

#TODO 
KIITI360_DETECTION_NAME_DICT = {
    "truck": DetectionType.VEHICLE,
    "bus": DetectionType.VEHICLE,
    "car": DetectionType.VEHICLE,
    "motorcycle": DetectionType.BICYCLE,
    "bicycle": DetectionType.BICYCLE,
    "pedestrian": DetectionType.PEDESTRIAN,
}

KITTI3602NUPLAN_IMU_CALIBRATION = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

KITTI3602NUPLAN_LIDAR_CALIBRATION = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def create_token(input_data: str) -> str:
    # TODO: Refactor this function.
    # TODO: Add a general function to create tokens from arbitrary data.
    if isinstance(input_data, str):
        input_data = input_data.encode("utf-8")

    hash_obj = hashlib.sha256(input_data)
    return hash_obj.hexdigest()[:16]


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

        def _has_modality(seq_name: str, modality_name: str, root: Path) -> bool:
            if modality_name == DIR_3D_BBOX:
                # expected: data_3d_bboxes/train/<seq_name>.xml
                xml_path = root / f"{seq_name}.xml"
                return xml_path.exists()
            else:
                return (root / seq_name).exists()

        valid_seqs: List[Path] = []
        for seq_dir in candidates:
            seq_name = seq_dir.name
            missing_modalities = [
                modality_name
                for modality_name, root in KITTI360_REQUIRED_MODALITY_ROOTS.items()
                if not _has_modality(seq_name, modality_name, root)
            ]
            if not missing_modalities:
                valid_seqs.append(seq_dir) #KITTI360_DATA_ROOT / DIR_2D_RAW /seq_name
            else:
                logging.info(
                    f"Sequence '{seq_name}' skipped: missing modalities {missing_modalities}. "
                    f"Root: {KITTI360_DATA_ROOT}"
                )
        logging.info(f"vadid sequences found: {valid_seqs}")
        return {"kitti360": valid_seqs}
    
    def get_available_splits(self) -> List[str]:
        """Returns a list of available raw data types."""
        return ["kitti360"]

    def convert_maps(self, worker: WorkerPool) -> None:
        logging.info("KITTI-360 does not provide standard maps. Skipping map conversion.")
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

            #TODO location
            metadata = LogMetadata(
                dataset="kitti360",
                log_name=log_name,
                location="None",
                timestep_seconds=KITTI360_DT,
                map_has_z=False,
            )

            vehicle_parameters = get_kitti360_station_wagon_parameters()
            camera_metadata = get_kitti360_camera_metadata()
            #TODO  now only velodyne lidar
            lidar_metadata = get_kitti360_lidar_metadata()

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
                for lidar_type in lidar_metadata.keys():
                    if data_converter_config.lidar_store_option == "path":
                        schema_column_list.append((lidar_type.serialize(), pa.string()))
                    elif data_converter_config.lidar_store_option == "binary":
                        raise NotImplementedError("Binary lidar storage is not implemented.")

            if data_converter_config.camera_store_option is not None:
                for camera_type in camera_metadata.keys():
                    if data_converter_config.camera_store_option == "path":
                        schema_column_list.append((camera_type.serialize(), pa.string()))
                        schema_column_list.append(
                            (f"{camera_type.serialize()}_extrinsic", pa.list_(pa.float64(), 4 * 4))
                        )
                    elif data_converter_config.camera_store_option == "binary":
                        raise NotImplementedError("Binary camera storage is not implemented.")

            recording_schema = pa.schema(schema_column_list)
            recording_schema = recording_schema.with_metadata(
                {
                    "log_metadata": json.dumps(asdict(metadata)),
                    "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                    "camera_metadata": camera_metadata_dict_to_json(camera_metadata),
                    "lidar_metadata": lidar_metadata_dict_to_json(lidar_metadata),
                }
            )

            _write_recording_table(log_name, recording_schema, log_file_path, data_converter_config)

        gc.collect()
    return []


def get_kitti360_camera_metadata() -> Dict[CameraType, CameraMetadata]:
    
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
        log_cam_infos[cam_type] = CameraMetadata(
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

def get_kitti360_lidar_metadata() -> Dict[LiDARType, LiDARMetadata]:
    metadata: Dict[LiDARType, LiDARMetadata] = {}

    cam2pose_txt = PATH_CALIB_ROOT / "calib_cam_to_pose.txt"
    if not cam2pose_txt.exists():
        raise FileNotFoundError(f"calib_cam_to_pose.txt file not found: {cam2pose_txt}")
    
    cam2velo_txt = PATH_CALIB_ROOT / "calib_cam_to_velo.txt"
    if not cam2velo_txt.exists():
        raise FileNotFoundError(f"calib_cam_to_velo.txt file not found: {cam2velo_txt}")
    
    lastrow = np.array([0,0,0,1]).reshape(1,4)

    with open(cam2pose_txt, 'r') as f:
        image_00 = next(f)
        values = list(map(float, image_00.strip().split()[1:]))
        matrix = np.array(values).reshape(3, 4)
        cam2pose = np.concatenate((matrix, lastrow))
        cam2pose = KITTI3602NUPLAN_IMU_CALIBRATION @ cam2pose
    
    cam2velo = np.concatenate((np.loadtxt(cam2velo_txt).reshape(3,4), lastrow))
    cam2velo = KITTI3602NUPLAN_LIDAR_CALIBRATION @ cam2velo

    extrinsic =  cam2velo @ np.linalg.inv(cam2pose)

    metadata[LiDARType.LIDAR_TOP] = LiDARMetadata(
        lidar_type=LiDARType.LIDAR_TOP,
        lidar_index=Kitti360LidarIndex,
        #TODO extrinsic needed to be same with nuplan
        extrinsic=extrinsic, 
    )
    return metadata

def _write_recording_table(
    log_name: str,
    recording_schema: pa.Schema,
    log_file_path: Path,
    data_converter_config: DataConverterConfig
) -> None:
    
    ts_list = _read_timestamps(log_name)
    ego_state_all = _extract_ego_state_all(log_name)
    detections_states,detections_velocity,detections_tokens,detections_types = _extract_detections(log_name,len(ts_list))

    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:
            for idx, tp in enumerate(ts_list):

                row_data = {
                    "token": [create_token(f"{log_name}_{idx}")],
                    "timestamp": [tp.time_us],
                    "detections_state": [detections_states[idx]],
                    "detections_velocity": [detections_velocity[idx]],
                    "detections_token": [detections_tokens[idx]],
                    "detections_type": [detections_types[idx]],
                    "ego_states": [ego_state_all[idx]],
                    "traffic_light_ids": [[]],
                    #may TODO traffic light types
                    "traffic_light_types": [[]],
                    "scenario_tag": [['unknown']],
                    "route_lane_group_ids": [[]],
                }

                if data_converter_config.lidar_store_option is not None:
                    lidar_data_dict = _extract_lidar(log_name, idx, data_converter_config)
                    for lidar_type, lidar_data in lidar_data_dict.items():
                        if lidar_data is not None:
                            row_data[lidar_type.serialize()] = [lidar_data]
                        else:
                            row_data[lidar_type.serialize()] = [None]

                if data_converter_config.camera_store_option is not None:
                    camera_data_dict = _extract_cameras(log_name, idx, data_converter_config)
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

#TODO default timestamps  and Synchronization all other sequences 
def _read_timestamps(log_name: str) -> Optional[List[TimePoint]]:
    # unix
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

def _extract_ego_state_all(log_name: str) -> List[List[float]]:

    ego_state_all: List[List[float]] = []

    pose_file = PATH_POSES_ROOT / log_name / "poses.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    poses = np.loadtxt(pose_file)
    poses_time = poses[:, 0] - 1  # Adjusting time to start from 0
    
    #TODO 
    oxts_path = Path("/data/jbwang/d123/data_poses/") / log_name / "oxts" / "data" 
    
    for idx in range(len(list(oxts_path.glob("*.txt")))):
        oxts_path_file = oxts_path / f"{int(idx):010d}.txt"
        oxts_data = np.loadtxt(oxts_path_file)

        roll, pitch, yaw = oxts_data[3:6]
        vehicle_parameters = get_kitti360_station_wagon_parameters()

        pos = np.searchsorted(poses_time, idx, side='right') - 1
        
        rear_axle_pose = StateSE3(
            x=poses[pos, 4],
            y=poses[pos, 8],
            z=poses[pos, 12],
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        # NOTE: The height to rear axle is not provided the dataset and is merely approximated.
        center = rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_pose, vehicle_parameters=vehicle_parameters)
        dynamic_state = DynamicStateSE3(
            velocity=Vector3D(
                x=oxts_data[8],
                y=oxts_data[9],
                z=oxts_data[10],
            ),
            acceleration=Vector3D(
                x=oxts_data[14],
                y=oxts_data[15],
                z=oxts_data[16],
            ),
            angular_velocity=Vector3D( 
                x=oxts_data[20],
                y=oxts_data[21],
                z=oxts_data[22],
            ),
        )
        ego_state_all.append(
                EgoStateSE3(
                center_se3=center,
                dynamic_state_se3=dynamic_state,
                vehicle_parameters=vehicle_parameters,
                timepoint=None,
            ).array.tolist()
        )
    return ego_state_all

#TODO now only divided by data_3d_semantics
# We may distinguish between image and lidar detections
# besides, now it is based only on start and end frame 
def _extract_detections(
    log_name: str,
    ts_len: int
) -> Tuple[List[List[float]], List[List[float]], List[str], List[int]]:
   
    detections_states: List[List[List[float]]] = [[] for _ in range(ts_len)]
    detections_velocity: List[List[List[float]]] = [[] for _ in range(ts_len)]
    detections_tokens: List[List[str]] = [[] for _ in range(ts_len)]
    detections_types: List[List[int]] = [[] for _ in range(ts_len)]

    bbox_3d_path = PATH_3D_BBOX_ROOT / "train" / f"{log_name}.xml"
    if not bbox_3d_path.exists():
        raise FileNotFoundError(f"BBox 3D file not found: {bbox_3d_path}")
    
    tree = ET.parse(bbox_3d_path)
    root = tree.getroot()

    for child in root:
        label = child.find('label').text
        if child.find('transform') is None or label not in KIITI360_DETECTION_NAME_DICT.keys():
            continue
        obj = KITTI360Bbox3D()
        obj.parseBbox(child)
        
        # static
        if obj.timestamp == -1:
            start_frame = obj.start_frame
            end_frame = obj.end_frame
            for frame in range(start_frame, end_frame + 1):
                #TODO check if valid in each frame
                if frame < 0 or frame >= ts_len:
                    continue
                #TODO  check yaw
                detections_states[frame].append(obj.get_state_array())
                detections_velocity[frame].append([0.0, 0.0, 0.0])
                detections_tokens[frame].append(str(obj.globalID))
                detections_types[frame].append(int(KIITI360_DETECTION_NAME_DICT[label]))
        # dynamic
        else:
            frame = obj.timestamp
            detections_states[frame].append(obj.get_state_array())
            #TODO velocity not provided
            detections_velocity[frame].append([0.0, 0.0, 0.0])
            detections_tokens[frame].append(str(obj.globalID))
            detections_types[frame].append(int(KIITI360_DETECTION_NAME_DICT[label]))

    return detections_states, detections_velocity, detections_tokens, detections_types

#TODO lidar extraction now only velo
def _extract_lidar(log_name: str, idx: int, data_converter_config: DataConverterConfig) -> Dict[LiDARType, Optional[str]]:
    lidar: Optional[str] = None
    lidar_full_path = PATH_3D_RAW_ROOT / log_name / "velodyne_points" / "data" / f"{idx:010d}.bin"
    if lidar_full_path.exists():
        if data_converter_config.lidar_store_option == "path":
            lidar = f"/data_3d_raw/{log_name}/velodyne_points/data/{idx:010d}.bin"
        elif data_converter_config.lidar_store_option == "binary":
            raise NotImplementedError("Binary lidar storage is not implemented.")
    else:
        raise FileNotFoundError(f"LiDAR file not found: {lidar_full_path}")
    return {LiDARType.LIDAR_TOP: lidar}

#TODO check camera extrinsic now is from camera to pose
def _extract_cameras(
    log_name: str, idx: int, data_converter_config: DataConverterConfig
) -> Dict[CameraType, Optional[str]]:
    
    camera_dict: Dict[str, Union[str, bytes]] = {}
    for camera_type, cam_dir_name in KITTI360_CAMERA_TYPES.items():
        img_path_png = PATH_2D_RAW_ROOT / log_name / cam_dir_name / "data_rect" / f"{idx:010d}.png"
        if img_path_png.exists():
            
            cam2pose_txt = PATH_CALIB_ROOT / "calib_cam_to_pose.txt"
            if not cam2pose_txt.exists():
                raise FileNotFoundError(f"calib_cam_to_pose.txt file not found: {cam2pose_txt}")
        
            lastrow = np.array([0,0,0,1]).reshape(1,4)

            with open(cam2pose_txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    key = parts[0][:-1]
                    if key == cam_dir_name:
                        values = list(map(float, parts[1:]))
                        matrix = np.array(values).reshape(3, 4)
                        cam2pose = np.concatenate((matrix, lastrow))
                        cam2pose = KITTI3602NUPLAN_IMU_CALIBRATION @ cam2pose

            if data_converter_config.camera_store_option == "path":
                camera_data = str(img_path_png), cam2pose.flatten().tolist()
            elif data_converter_config.camera_store_option == "binary":
                with open(img_path_png, "rb") as f:
                    camera_data = f.read(), cam2pose
        else:
            raise FileNotFoundError(f"Camera image not found: {img_path_png}")
        camera_dict[camera_type] = camera_data
    return camera_dict
