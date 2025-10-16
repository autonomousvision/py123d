import os
import re
import yaml
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pickle
from collections import defaultdict
import datetime
import xml.etree.ElementTree as ET
import logging
from pyquaternion import Quaternion

from d123.common.multithreading.worker_utils import WorkerPool, worker_map

from d123.datatypes.detections.detection import (
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionWrapper,
)
from d123.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from d123.datatypes.sensors.camera.fisheye_mei_camera import (
    FisheyeMEICameraMetadata,
    FisheyeMEICameraType,
    FisheyeMEIDistortion,
    FisheyeMEIProjection,
)
from d123.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
from d123.conversion.utils.sensor_utils.lidar_index_registry import Kitti360LidarIndex
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3, EgoStateSE3Index
from d123.datatypes.vehicle_state.vehicle_parameters import get_kitti360_station_wagon_parameters,rear_axle_se3_to_center_se3
from d123.common.utils.uuid import create_deterministic_uuid
from d123.conversion.abstract_dataset_converter import AbstractDatasetConverter
from d123.conversion.dataset_converter_config import DatasetConverterConfig
from d123.conversion.log_writer.abstract_log_writer import AbstractLogWriter
from d123.conversion.log_writer.arrow_log_writer import ArrowLogWriter
from d123.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from d123.datatypes.maps.map_metadata import MapMetadata
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.conversion.datasets.kitti_360.kitti_360_helper import KITTI360Bbox3D, KITTI3602NUPLAN_IMU_CALIBRATION, get_lidar_extrinsic
from d123.conversion.datasets.kitti_360.labels import KITTI360_DETECTION_NAME_DICT, kittiId2label, BBOX_LABLES_TO_DETECTION_NAME_DICT
from d123.conversion.datasets.kitti_360.kitti_360_map_conversion import (
    convert_kitti360_map_with_writer
)
from d123.geometry import BoundingBoxSE3, BoundingBoxSE3Index, StateSE3, Vector3D, Vector3DIndex
from d123.geometry.rotation import EulerAngles

KITTI360_DT: Final[float] = 0.1

KITTI360_DATA_ROOT = Path(os.environ["KITTI360_DATA_ROOT"])

KITTI360_CAMERA_TYPES = {
    PinholeCameraType.CAM_STEREO_L: "image_00",  
    PinholeCameraType.CAM_STEREO_R: "image_01",   
    FisheyeMEICameraType.CAM_L: "image_02", 
    FisheyeMEICameraType.CAM_R: "image_03", 
}

DIR_2D_RAW = "data_2d_raw"
DIR_2D_SMT = "data_2d_semantics"
DIR_3D_RAW = "data_3d_raw"
DIR_3D_SMT = "data_3d_semantics"
DIR_3D_BBOX = "data_3d_bboxes"
DIR_POSES = "data_poses"
DIR_CALIB = "calibration"

# PATH_2D_RAW_ROOT: Path = KITTI360_DATA_ROOT / DIR_2D_RAW
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

D123_DEVKIT_ROOT = Path(os.environ["D123_DEVKIT_ROOT"])
PREPOCESS_DETECTION_DIR = D123_DEVKIT_ROOT / "d123" / "conversion" / "datasets" / "kitti_360" / "detection_preprocess"

def create_token(split: str, log_name: str, timestamp_us: int, misc: str = None) -> str:
    """Create a deterministic UUID-based token for KITTI-360 data.
    
    :param split: The data split (e.g., "kitti360")
    :param log_name: The name of the log without file extension
    :param timestamp_us: The timestamp in microseconds
    :param misc: Any additional information to include in the UUID, defaults to None
    :return: The generated deterministic UUID as hex string
    """
    uuid_obj = create_deterministic_uuid(split=split, log_name=log_name, timestamp_us=timestamp_us, misc=misc)
    return uuid_obj.hex

def get_kitti360_map_metadata(split: str, log_name: str) -> MapMetadata:
    return MapMetadata(
        dataset="kitti360",
        split=split,
        log_name=log_name,
        location=log_name,
        map_has_z=True,
        map_is_local=True,
    )

class Kitti360DataConverter(AbstractDatasetConverter):
    def __init__(
        self,
        splits: List[str],
        log_path: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        super().__init__(dataset_converter_config)
        for split in splits:
            assert (
                split in self.get_available_splits()
            ), f"Split {split} is not available. Available splits: {self.available_splits}"

        self._splits: List[str] = splits
        self._log_path: Path = Path(log_path)
        self._log_paths_and_split: List[Tuple[Path, str]] = self._collect_log_paths()
        
        self._total_maps = len(self._log_paths_and_split)  # Each log has its own map
        self._total_logs = len(self._log_paths_and_split)

    def _collect_log_paths(self) -> List[Tuple[Path, str]]:
        """
        Collect candidate sequence folders under data_2d_raw that end with '_sync',
        and keep only those sequences that are present in ALL required modality roots
        (e.g., data_2d_semantics, data_3d_raw, etc.).
        Returns a list of (log_path, split) tuples.
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

        log_paths_and_split: List[Tuple[Path, str]] = []
        for seq_dir in candidates:
            seq_name = seq_dir.name
            missing_modalities = [
                modality_name
                for modality_name, root in KITTI360_REQUIRED_MODALITY_ROOTS.items()
                if not _has_modality(seq_name, modality_name, root)
            ]
            if not missing_modalities:
                log_paths_and_split.append((seq_dir, "kitti360"))
            else:
                logging.info(
                    f"Sequence '{seq_name}' skipped: missing modalities {missing_modalities}. "
                    f"Root: {KITTI360_DATA_ROOT}"
                )
        
        logging.info(f"Valid sequences found: {len(log_paths_and_split)}")
        return log_paths_and_split
    
    def get_available_splits(self) -> List[str]:
        """Returns a list of available raw data types."""
        return ["kitti360"]
    
    def get_number_of_maps(self) -> int:
        """Returns the number of available raw data maps for conversion."""
        return self._total_maps
    
    def get_number_of_logs(self) -> int:
        """Returns the number of available raw data logs for conversion."""
        return self._total_logs
    
    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """
        Convert a single map in raw data format to the uniform 123D format.
        :param map_index: The index of the map to convert.
        :param map_writer: The map writer to use for writing the converted map.
        """
        source_log_path, split = self._log_paths_and_split[map_index]
        log_name = source_log_path.stem
        
        map_metadata = get_kitti360_map_metadata(split, log_name)
        
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)
        if map_needs_writing:
            convert_kitti360_map_with_writer(log_name, map_writer)
        
        map_writer.close()
    
    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """
        Convert a single log in raw data format to the uniform 123D format.
        :param log_index: The index of the log to convert.
        :param log_writer: The log writer to use for writing the converted log.
        """
        source_log_path, split = self._log_paths_and_split[log_index]
        log_name = source_log_path.stem
        
        # Create log metadata
        log_metadata = LogMetadata(
            dataset="kitti360",
            split=split,
            log_name=log_name,
            location=log_name,
            timestep_seconds=KITTI360_DT,
            vehicle_parameters=get_kitti360_station_wagon_parameters(),
            camera_metadata=get_kitti360_camera_metadata(),
            lidar_metadata=get_kitti360_lidar_metadata(),
            map_metadata=get_kitti360_map_metadata(split, log_name)
        )
        
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)
        if log_needs_writing:
            _write_recording_table(log_name, log_writer, self.dataset_converter_config)
        
        log_writer.close()

def get_kitti360_camera_metadata() -> Dict[Union[PinholeCameraType, FisheyeMEICameraType], Union[PinholeCameraMetadata, FisheyeMEICameraMetadata]]:
    
    persp = PATH_CALIB_ROOT / "perspective.txt"

    assert persp.exists()
    persp_result = {"image_00": {}, "image_01": {}}

    with open(persp, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            key, value = ln.split(" ", 1)
            cam_id = key.split("_")[-1][:2]
            if key.startswith("P_rect_"):
                persp_result[f"image_{cam_id}"]["intrinsic"] = _read_projection_matrix(ln)
            elif key.startswith("S_rect_"):
                persp_result[f"image_{cam_id}"]["wh"] = [int(round(float(x))) for x in value.split()]
            elif key.startswith("D_"):
                persp_result[f"image_{cam_id}"]["distortion"] = [float(x) for x in value.split()]
    
    fisheye_camera02_path = PATH_CALIB_ROOT / "image_02.yaml"
    fisheye_camera03_path = PATH_CALIB_ROOT / "image_03.yaml"
    assert fisheye_camera02_path.exists() and fisheye_camera03_path.exists()
    fisheye02 = _readYAMLFile(fisheye_camera02_path)
    fisheye03 = _readYAMLFile(fisheye_camera03_path)
    fisheye_result = {"image_02": fisheye02, "image_03": fisheye03}
    
    log_cam_infos: Dict[Union[PinholeCameraType, FisheyeMEICameraType], Union[PinholeCameraMetadata, FisheyeMEICameraMetadata]] = {}
    for cam_type, cam_name in KITTI360_CAMERA_TYPES.items():
        if cam_name in ["image_00", "image_01"]:
            log_cam_infos[cam_type] = PinholeCameraMetadata(
                camera_type=cam_type,
                width=persp_result[cam_name]["wh"][0],
                height=persp_result[cam_name]["wh"][1],
                intrinsics=PinholeIntrinsics.from_camera_matrix(np.array(persp_result[cam_name]["intrinsic"])),
                distortion=PinholeDistortion.from_array(np.array(persp_result[cam_name]["distortion"])),
            )
        elif cam_name in ["image_02","image_03"]:
            distortion_params = fisheye_result[cam_name]["distortion_parameters"]
            distortion = FisheyeMEIDistortion(
                k1=distortion_params['k1'],
                k2=distortion_params['k2'],
                p1=distortion_params['p1'],
                p2=distortion_params['p2'],
            )
            
            projection_params = fisheye_result[cam_name]["projection_parameters"]
            projection = FisheyeMEIProjection(
                gamma1=projection_params['gamma1'],
                gamma2=projection_params['gamma2'],
                u0=projection_params['u0'],
                v0=projection_params['v0'],
            )
            
            log_cam_infos[cam_type] = FisheyeMEICameraMetadata(
                camera_type=cam_type,
                width=fisheye_result[cam_name]["image_width"],
                height=fisheye_result[cam_name]["image_height"],
                mirror_parameter=fisheye_result[cam_name]["mirror_parameters"],
                distortion=distortion,
                projection=projection,
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

def _readYAMLFile(fileName:Path) -> Dict[str, Any]:
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.safe_load(yamlFileOut)
    return ret

def get_kitti360_lidar_metadata() -> Dict[LiDARType, LiDARMetadata]:
    metadata: Dict[LiDARType, LiDARMetadata] = {}
    extrinsic = get_lidar_extrinsic()
    extrinsic_state_se3 = StateSE3.from_transformation_matrix(extrinsic)
    metadata[LiDARType.LIDAR_TOP] = LiDARMetadata(
        lidar_type=LiDARType.LIDAR_TOP,
        lidar_index=Kitti360LidarIndex,
        extrinsic=extrinsic_state_se3, 
    )
    return metadata

def _write_recording_table(
    log_name: str,
    log_writer: AbstractLogWriter,
    data_converter_config: DatasetConverterConfig
) -> None:
    
    ts_list: List[TimePoint] = _read_timestamps(log_name)
    ego_state_all, valid_timestamp = _extract_ego_state_all(log_name)
    ego_states_xyz = np.array([ego_state.center.array[:3] for ego_state in ego_state_all],dtype=np.float64)
    box_detection_wrapper_all = _extract_detections(log_name,len(ts_list),ego_states_xyz,valid_timestamp)
    logging.info(f"Number of valid timestamps with ego states: {len(valid_timestamp)}")
    for idx in range(len(valid_timestamp)):
        valid_idx = valid_timestamp[idx]
         
        cameras = _extract_cameras(log_name, valid_idx, data_converter_config)
        lidars = _extract_lidar(log_name, valid_idx, data_converter_config)

        log_writer.write(
            timestamp=ts_list[valid_idx],
            ego_state=ego_state_all[idx],
            box_detections=box_detection_wrapper_all[valid_idx],
            traffic_lights=None,
            cameras=cameras,
            lidars=lidars,
            scenario_tags=None,
            route_lane_group_ids=None,
        )

    # if SORT_BY_TIMESTAMP:
    #     recording_table = open_arrow_table(log_file_path)
    #     recording_table = recording_table.sort_by([("timestamp", "ascending")])
    #     write_arrow_table(recording_table, log_file_path)

def _read_timestamps(log_name: str) -> Optional[List[TimePoint]]:
    """
    Read KITTI-360 timestamps for the given sequence and return Unix epoch timestamps.
    """
    ts_files = [
        PATH_3D_RAW_ROOT / log_name / "velodyne_points" / "timestamps.txt",
        PATH_2D_RAW_ROOT / log_name / "image_00" / "timestamps.txt",
        PATH_2D_RAW_ROOT / log_name / "image_01" / "timestamps.txt",
    ]
    
    if log_name == "2013_05_28_drive_0002_sync":
        ts_files = ts_files[1:]

    for ts_file in ts_files:
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

def _extract_ego_state_all(log_name: str) -> Tuple[List[EgoStateSE3], List[int]]:

    ego_state_all: List[List[float]] = []

    pose_file = PATH_POSES_ROOT / log_name / "poses.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    poses = np.loadtxt(pose_file)
    poses_time = poses[:, 0].astype(np.int32)
    valid_timestamp: List[int] = list(poses_time)
     
    oxts_path =  PATH_POSES_ROOT / log_name / "oxts" / "data" 
    
    for idx in range(len(valid_timestamp)):
        oxts_path_file = oxts_path / f"{int(valid_timestamp[idx]):010d}.txt"
        oxts_data = np.loadtxt(oxts_path_file)

        vehicle_parameters = get_kitti360_station_wagon_parameters()

        pos = idx 
        if log_name=="2013_05_28_drive_0004_sync" and pos == 0:
            pos = 1
        
        # NOTE you can use oxts_data[3:6] as roll, pitch, yaw for simplicity
        #roll, pitch, yaw = oxts_data[3:6]
        r00, r01, r02 = poses[pos, 1:4]
        r10, r11, r12 = poses[pos, 5:8]
        r20, r21, r22 = poses[pos, 9:12]
        R_mat = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]], dtype=np.float64)
        R_mat_cali = R_mat @ KITTI3602NUPLAN_IMU_CALIBRATION[:3,:3]
        yaw, pitch, roll = Quaternion(matrix=R_mat_cali[:3, :3]).yaw_pitch_roll

        ego_quaternion = EulerAngles(roll=roll, pitch=pitch, yaw=yaw).quaternion
        rear_axle_pose = StateSE3(
            x=poses[pos, 4],
            y=poses[pos, 8],
            z=poses[pos, 12],
            qw=ego_quaternion.qw,
            qx=ego_quaternion.qx,
            qy=ego_quaternion.qy,
            qz=ego_quaternion.qz,
        )

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
            )
        )
    return ego_state_all, valid_timestamp

def _extract_detections(
    log_name: str,
    ts_len: int,
    ego_states_xyz: np.ndarray,
    valid_timestamp: List[int],
) -> List[BoxDetectionWrapper]:
   
    detections_states: List[List[List[float]]] = [[] for _ in range(ts_len)]
    detections_velocity: List[List[List[float]]] = [[] for _ in range(ts_len)]
    detections_tokens: List[List[str]] = [[] for _ in range(ts_len)]
    detections_types: List[List[int]] = [[] for _ in range(ts_len)]

    if log_name == "2013_05_28_drive_0004_sync":
        bbox_3d_path = PATH_3D_BBOX_ROOT / "train_full" / f"{log_name}.xml"
    else:
        bbox_3d_path = PATH_3D_BBOX_ROOT / "train" / f"{log_name}.xml"
    if not bbox_3d_path.exists():
        raise FileNotFoundError(f"BBox 3D file not found: {bbox_3d_path}")
    
    tree = ET.parse(bbox_3d_path)
    root = tree.getroot()

    detection_preprocess_path = PREPOCESS_DETECTION_DIR / f"{log_name}_detection_preprocessed.pkl"
    if detection_preprocess_path.exists():
        with open(detection_preprocess_path, "rb") as f:
            detection_preprocess_result = pickle.load(f)
            static_records_dict = {record_item["global_id"]: record_item for record_item in detection_preprocess_result["static"]}
            logging.info(f"Loaded detection preprocess data from {detection_preprocess_path}")
    else:
        detection_preprocess_result = None

    dynamic_objs: Dict[int, List[KITTI360Bbox3D]] = defaultdict(list)

    for child in root:
        if child.find('semanticId') is not None:
            semanticIdKITTI = int(child.find('semanticId').text)
            name = kittiId2label[semanticIdKITTI].name
        else:
            lable = child.find('label').text
            name = BBOX_LABLES_TO_DETECTION_NAME_DICT.get(lable, 'unknown')
        if child.find('transform') is None or name not in KITTI360_DETECTION_NAME_DICT.keys():
            continue
        obj = KITTI360Bbox3D()
        obj.parseBbox(child)
        
        #static object
        if obj.timestamp == -1:
            if detection_preprocess_result is None:
                obj.filter_by_radius(ego_states_xyz,valid_timestamp,radius=50.0)
            else:
                obj.load_detection_preprocess(static_records_dict)
            for record in obj.valid_frames["records"]:
                frame = record["timestamp"]
                detections_states[frame].append(obj.get_state_array())
                detections_velocity[frame].append(np.array([0.0, 0.0, 0.0]))
                detections_tokens[frame].append(str(obj.globalID))
                detections_types[frame].append(KITTI360_DETECTION_NAME_DICT[obj.name])  
        else:
            global_ID = obj.globalID
            dynamic_objs[global_ID].append(obj)

    # dynamic object
    for global_id, obj_list in dynamic_objs.items():
        obj_list.sort(key=lambda obj: obj.timestamp)
        num_frames = len(obj_list)
        
        positions = [obj.get_state_array()[:3] for obj in obj_list]
        timestamps = [int(obj.timestamp) for obj in obj_list]

        velocities = []

        for i in range(1, num_frames - 1):
            dt_frames = timestamps[i+1] - timestamps[i-1]
            if dt_frames > 0:
                dt = dt_frames * KITTI360_DT
                vel = (positions[i+1] - positions[i-1]) / dt
                vel = KITTI3602NUPLAN_IMU_CALIBRATION[:3,:3] @ obj_list[i].Rm.T @ vel
            else:
                vel = np.zeros(3)
            velocities.append(vel)
        
        if num_frames > 1:
            # first and last frame
            velocities.insert(0, velocities[0])
            velocities.append(velocities[-1])
        elif num_frames == 1:
            velocities.append(np.zeros(3))

        for obj, vel in zip(obj_list, velocities):
            frame = obj.timestamp
            detections_states[frame].append(obj.get_state_array())
            detections_velocity[frame].append(vel)
            detections_tokens[frame].append(str(obj.globalID))
            detections_types[frame].append(KITTI360_DETECTION_NAME_DICT[obj.name])

    box_detection_wrapper_all: List[BoxDetectionWrapper] = []
    for frame in range(ts_len):
        box_detections: List[BoxDetectionSE3] = []
        for state, velocity, token, detection_type in zip(
            detections_states[frame],
            detections_velocity[frame],
            detections_tokens[frame],
            detections_types[frame],
        ):
            if state is None:
                break
            detection_metadata = BoxDetectionMetadata(
                detection_type=detection_type,
                timepoint=None,
                track_token=token,
                confidence=None,
            )
            bounding_box_se3 = BoundingBoxSE3.from_array(state)
            velocity_vector = Vector3D.from_array(velocity)
            box_detection = BoxDetectionSE3(
                metadata=detection_metadata,
                bounding_box_se3=bounding_box_se3,
                velocity=velocity_vector,
            )
            box_detections.append(box_detection)
        box_detection_wrapper_all.append(BoxDetectionWrapper(box_detections=box_detections))
    return box_detection_wrapper_all

def _extract_lidar(log_name: str, idx: int, data_converter_config: DatasetConverterConfig) -> Dict[LiDARType, Optional[str]]:
    
    #NOTE special case for sequence 2013_05_28_drive_0002_sync which has no lidar data before frame 4391
    if log_name == "2013_05_28_drive_0002_sync" and idx <= 4390:
        return {LiDARType.LIDAR_TOP: None}
    
    lidar: Optional[str] = None
    lidar_full_path = PATH_3D_RAW_ROOT / log_name / "velodyne_points" / "data" / f"{idx:010d}.bin"
    if lidar_full_path.exists():
        if data_converter_config.lidar_store_option == "path":
            lidar = f"data_3d_raw/{log_name}/velodyne_points/data/{idx:010d}.bin"
        elif data_converter_config.lidar_store_option == "binary":
            raise NotImplementedError("Binary lidar storage is not implemented.")
    else:
        raise FileNotFoundError(f"LiDAR file not found: {lidar_full_path}")
    return {LiDARType.LIDAR_TOP: lidar}

def _extract_cameras(
    log_name: str, idx: int, data_converter_config: DatasetConverterConfig
) -> Dict[Union[PinholeCameraType, FisheyeMEICameraType], Optional[Tuple[Union[str, bytes], StateSE3]]]:
    
    camera_dict: Dict[Union[PinholeCameraType, FisheyeMEICameraType], Optional[Tuple[Union[str, bytes], StateSE3]]] = {}
    for camera_type, cam_dir_name in KITTI360_CAMERA_TYPES.items():
        if cam_dir_name in ["image_00", "image_01"]:
            img_path_png = PATH_2D_RAW_ROOT / log_name / cam_dir_name / "data_rect" / f"{idx:010d}.png"
        elif cam_dir_name in ["image_02", "image_03"]:
            img_path_png = PATH_2D_RAW_ROOT / log_name / cam_dir_name / "data_rgb" / f"{idx:010d}.png"

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

        if img_path_png.exists():
            if data_converter_config.camera_store_option == "path":
                camera_data = str(img_path_png)
            elif data_converter_config.camera_store_option == "binary":
                with open(img_path_png, "rb") as f:
                    camera_data = f.read()
        else:
            camera_data = None
        
        camera_extrinsic = StateSE3.from_transformation_matrix(cam2pose)
        camera_dict[camera_type] = camera_data, camera_extrinsic
    return camera_dict
