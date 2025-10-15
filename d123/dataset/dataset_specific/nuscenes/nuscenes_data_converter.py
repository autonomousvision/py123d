import gc
import json
import os
import pickle
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.prediction import PredictHelper
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from nuplan.planning.utils.multithreading.worker_utils import worker_map
from d123.script.builders.worker_pool_builder import WorkerPool
from d123.common.datatypes.detection.detection import TrafficLightStatus
from d123.common.datatypes.detection.detection_types import DetectionType
from d123.common.datatypes.sensor.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar import LiDARMetadata, LiDARType, lidar_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar_index import NuscenesLidarIndex
from d123.common.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3, EgoStateSE3Index
from d123.common.datatypes.vehicle_state.vehicle_parameters import (
    get_nuplan_pacifica_parameters,
    rear_axle_se3_to_center_se3,
)
from d123.common.geometry.base import StateSE3
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3, BoundingBoxSE3Index
from d123.common.geometry.vector import Vector3D, Vector3DIndex
from d123.dataset.arrow.helper import open_arrow_table, write_arrow_table
from d123.dataset.dataset_specific.nuscenes.nuscenes_map_conversion import NUSCENES_MAPS, NuscenesMapConverter
from d123.dataset.dataset_specific.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.dataset.logs.log_metadata import LogMetadata

TARGET_DT: Final[float] = 0.1
NUSCENES_DT: Final[float] = 0.5
SORT_BY_TIMESTAMP: Final[bool] = True
NUSCENES_DETECTION_NAME_DICT = {
    # Vehicles (4+ wheels)
    "vehicle.car": DetectionType.VEHICLE,
    "vehicle.truck": DetectionType.VEHICLE,
    "vehicle.bus.bendy": DetectionType.VEHICLE,
    "vehicle.bus.rigid": DetectionType.VEHICLE,
    "vehicle.construction": DetectionType.VEHICLE,
    "vehicle.emergency.ambulance": DetectionType.VEHICLE,
    "vehicle.emergency.police": DetectionType.VEHICLE,
    "vehicle.trailer": DetectionType.VEHICLE,

    # Bicycles / Motorcycles
    "vehicle.bicycle": DetectionType.BICYCLE,
    "vehicle.motorcycle": DetectionType.BICYCLE,

    # Pedestrians (all subtypes)
    "human.pedestrian.adult": DetectionType.PEDESTRIAN,
    "human.pedestrian.child": DetectionType.PEDESTRIAN,
    "human.pedestrian.construction_worker": DetectionType.PEDESTRIAN,
    "human.pedestrian.personal_mobility": DetectionType.PEDESTRIAN,
    "human.pedestrian.police_officer": DetectionType.PEDESTRIAN,
    "human.pedestrian.stroller": DetectionType.PEDESTRIAN,
    "human.pedestrian.wheelchair": DetectionType.PEDESTRIAN,

    # Traffic cone / barrier
    "movable_object.trafficcone": DetectionType.TRAFFIC_CONE,
    "movable_object.barrier": DetectionType.BARRIER,

    # Generic objects
    "movable_object.pushable_pullable": DetectionType.GENERIC_OBJECT,
    "movable_object.debris": DetectionType.GENERIC_OBJECT,
    "static_object.bicycle_rack": DetectionType.GENERIC_OBJECT,
    "animal": DetectionType.GENERIC_OBJECT,
}

NUSCENES_CAMERA_TYPES = {
    CameraType.CAM_F0: "CAM_FRONT",
    CameraType.CAM_B0: "CAM_BACK",
    CameraType.CAM_L1: "CAM_FRONT_LEFT",
    CameraType.CAM_L2: "CAM_BACK_LEFT",
    CameraType.CAM_R1: "CAM_FRONT_RIGHT",
    CameraType.CAM_R2: "CAM_BACK_RIGHT",
}
NUSCENES_DATA_ROOT = Path(os.environ["NUSCENES_DATA_ROOT"])


class NuScenesDataConverter(RawDataConverter):
    def __init__(
            self,
            splits: List[str],
            nuscenes_data_root: Union[Path, str],
            data_converter_config: DataConverterConfig,
            version: str = "v1.0-trainval",
    ) -> None:
        super().__init__(data_converter_config)
        self._splits: List[str] = splits
        self._nuscenes_data_root: Path = Path(nuscenes_data_root)
        self._version = version
        self._scene_tokens_per_split: Dict[str, List[str]] = self._collect_scene_tokens()
        self._target_dt: float = TARGET_DT

    def _collect_scene_tokens(self) -> Dict[str, List[str]]:
        scene_tokens_per_split: Dict[str, List[str]] = {}
        nusc = NuScenes(version=self._version, dataroot=str(self._nuscenes_data_root), verbose=False)

        scene_splits = create_splits_scenes()
        available_scenes = [scene for scene in nusc.scene]

        for split in self._splits:
            # Map the split name to the division of nuScenes
            nusc_split = split.replace("nuscenes_", "")
            if nusc_split == "trainval":
                scene_names = scene_splits['train'] + scene_splits['val']
            else:
                scene_names = scene_splits.get(nusc_split, [])

            #get token
            scene_tokens = [
                scene['token'] for scene in available_scenes
                if scene['name'] in scene_names
            ]
            scene_tokens_per_split[split] = scene_tokens

        return scene_tokens_per_split

    def get_available_splits(self) -> List[str]:
        return [
            "nuscenes_train",
            "nuscenes_val",
            "nuscenes_test",
            "nuscenes_mini_train",
            "nuscenes_mini_val",
        ]

    def convert_maps(self, worker: WorkerPool) -> None:
        worker_map(
            worker,
            partial(convert_nuscenes_map_to_gpkg, data_converter_config=self.data_converter_config),
            list(NUSCENES_MAPS.keys()),
        )

    def convert_logs(self, worker: WorkerPool) -> None:
        log_args = [
            {
                "scene_token": scene_token,
                "split": split,
            }
            for split, scene_tokens in self._scene_tokens_per_split.items()
            for scene_token in scene_tokens
        ]

        worker_map(
            worker,
            partial(
                convert_nuscenes_log_to_arrow,
                data_converter_config=self.data_converter_config,
                version=self._version,
            ),
            log_args,
        )


def convert_nuscenes_map_to_gpkg(map_names: List[str], data_converter_config: DataConverterConfig) -> List[Any]:
    for map_name in map_names:
        map_path = data_converter_config.output_path / "maps" / f"nuscenes_{map_name}.gpkg"
        if data_converter_config.force_map_conversion or not map_path.exists():
            map_path.unlink(missing_ok=True)
            NuscenesMapConverter(data_converter_config.output_path / "maps").convert(map_name=map_name)
    return []



def convert_nuscenes_log_to_arrow(
        args: List[Dict[str, Union[str, List[str]]]],
        data_converter_config: DataConverterConfig,
        version: str
) -> List[Any]:
    for log_info in args:
        scene_token: str = log_info["scene_token"]
        split: str = log_info["split"]

        nusc = NuScenes(version=version, dataroot=str(NUSCENES_DATA_ROOT), verbose=False)
        scene = nusc.get("scene", scene_token)

        log_file_path = data_converter_config.output_path / split / f"{scene_token}.arrow"

        if data_converter_config.force_log_conversion or not log_file_path.exists():
            if not log_file_path.parent.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # 构建schema
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

            if data_converter_config.lidar_store_option == "path":
                schema_column_list.append(("lidar", pa.string()))

            if data_converter_config.camera_store_option == "path":
                for camera_type in NUSCENES_CAMERA_TYPES.keys():
                    schema_column_list.append((camera_type.serialize(), pa.string()))
                    schema_column_list.append((f"{camera_type.serialize()}_extrinsic", pa.list_(pa.float64(), 4*4)))

            recording_schema = pa.schema(schema_column_list)

            log_record = nusc.get("log", scene["log_token"])
            location = log_record["location"]
            metadata = LogMetadata(
                dataset="nuscenes",
                log_name=scene["name"],
                location=location,
                timestep_seconds=TARGET_DT,
                map_has_z=True,
            )
            vehicle_parameters = get_nuplan_pacifica_parameters()
            # camera
            camera_metadata = get_nuscenes_camera_metadata(nusc, scene)
            # lidar
            lidar_metadata = get_nuscenes_lidar_metadata(nusc, scene)

            recording_schema = recording_schema.with_metadata(
                {
                    "log_metadata": json.dumps(asdict(metadata)),
                    "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                    "camera_metadata": camera_metadata_dict_to_json(camera_metadata),
                    "lidar_metadata": lidar_metadata_dict_to_json(lidar_metadata),
                }
            )

            _write_recording_table(nusc, scene, recording_schema, log_file_path, data_converter_config)

        del nusc
        gc.collect()

    return []


def get_nuscenes_camera_metadata(nusc: NuScenes, scene: Dict[str, Any]) -> Dict[str, CameraMetadata]:
    log_cam_infos: Dict[str, CameraMetadata] = {}

    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)

    for camera_type, camera_channel in NUSCENES_CAMERA_TYPES.items():
        cam_token = first_sample["data"][camera_channel]
        cam_data = nusc.get("sample_data", cam_token)
        calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

        intrinsic = np.array(calib["camera_intrinsic"])
        translation = np.array(calib["translation"])
        rotation = Quaternion(calib["rotation"]).rotation_matrix
        distortion = np.zeros(5)  # nuScenes does not provide distortion parameters.

        log_cam_infos[camera_type] = CameraMetadata(
            camera_type=camera_type,
            width=cam_data["width"],
            height=cam_data["height"],
            intrinsic=intrinsic,
            distortion=distortion,
            translation=translation,
            rotation=rotation,
        )

    return log_cam_infos


def get_nuscenes_lidar_metadata(nusc: NuScenes, scene: Dict[str, Any]) -> Dict[LiDARType, LiDARMetadata]:
    # Obtain the LIDAR_TOP data of the first sample in the scene
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)
    lidar_token = first_sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    translation = np.array(calib["translation"])
    rotation = Quaternion(calib["rotation"]).rotation_matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation

    lidar_type = LiDARType.LIDAR_MERGED

    metadata = {}
    metadata[lidar_type] = LiDARMetadata(
        lidar_type=lidar_type,
        lidar_index=NuscenesLidarIndex,  
        extrinsic=extrinsic,
    )

    return metadata


def _write_recording_table(
        nusc: NuScenes,
        scene: Dict[str, Any],
        recording_schema: pa.schema,
        log_file_path: Path,
        data_converter_config: DataConverterConfig,
) -> None:
    can_bus = NuScenesCanBus(dataroot=str(NUSCENES_DATA_ROOT))
    helper = PredictHelper(nusc)

    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:
            step_interval = max(1, int(TARGET_DT / NUSCENES_DT))
            sample_count = 0

            sample_token = scene["first_sample_token"]
            while sample_token:
                if sample_count % step_interval == 0:
                    sample = nusc.get("sample", sample_token)

                    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

                    detections_state, detections_velocity, detections_token, detections_types = _extract_detections(
                        nusc, sample)

                    ego_state = _extract_ego_state(nusc, sample, can_bus)

                    # nuScenes do not have）
                    traffic_light_ids, traffic_light_types = [], []
                    #（nuScenes do not have）
                    route_lane_group_ids = []

                    # (nuScenes do not have）
                    scenario_tag = ["unknown"]

                    row_data = {
                        "token": [sample_token],
                        "timestamp": [sample["timestamp"]],
                        "detections_state": [detections_state],
                        "detections_velocity": [detections_velocity],
                        "detections_token": [detections_token],
                        "detections_type": [detections_types],
                        "ego_states": [ego_state],
                        "traffic_light_ids": [traffic_light_ids],
                        "traffic_light_types": [traffic_light_types],
                        "scenario_tag": [scenario_tag],
                        "route_lane_group_ids": [route_lane_group_ids],
                    }

                    #lidar
                    if data_converter_config.lidar_store_option == "path":
                        lidar_path = NUSCENES_DATA_ROOT / lidar_data["filename"]
                        row_data["lidar"] = [str(lidar_path)]

                    # camera
                    if data_converter_config.camera_store_option == "path":
                        for camera_type, camera_channel in NUSCENES_CAMERA_TYPES.items():
                            cam_token = sample["data"][camera_channel]
                            cam_data = nusc.get("sample_data", cam_token)
                            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
                            translation = np.array(calib["translation"])
                            rotation = Quaternion(calib["rotation"]).rotation_matrix
                            extrinsic = np.eye(4)
                            extrinsic[:3, :3] = rotation
                            extrinsic[:3, 3] = translation
                            extrinsic_list = extrinsic.flatten().tolist()

                            cam_path = NUSCENES_DATA_ROOT / cam_data["filename"]
                            row_data[camera_type.serialize()] = [str(cam_path)]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [extrinsic_list]

                    batch = pa.record_batch(row_data, schema=recording_schema)
                    writer.write_batch(batch)

                sample_token = sample["next"]
                sample_count += 1

    if SORT_BY_TIMESTAMP:
        recording_table = open_arrow_table(log_file_path)
        recording_table = recording_table.sort_by([("timestamp", "ascending")])
        write_arrow_table(recording_table, log_file_path)


def _extract_detections(
        nusc: NuScenes,
        sample: Dict[str, Any]
) -> Tuple[List[List[float]], List[List[float]], List[str], List[int]]:
    detections_state = []
    detections_velocity = []
    detections_token = []
    detections_types = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

        center = StateSE3(
            x=box.center[0],
            y=box.center[1],
            z=box.center[2],
            roll=0.0,  # nuScenes do not have roll
            pitch=0.0,  # nuScenes do not have pitch
            yaw=box.orientation.yaw_pitch_roll[0],
        )
        bounding_box = BoundingBoxSE3(center, box.wlh[1], box.wlh[0], box.wlh[2])

        velocity = nusc.box_velocity(ann_token)
        velocity_3d = [velocity[0], velocity[1], 0.0]  # 转换为3D

        category = ann["category_name"]
        det_type = None
        for key, value in NUSCENES_DETECTION_NAME_DICT.items():
            if category.startswith(key):
                det_type = value.value
                break

        if det_type is None:
            raise ValueError(f"Unmapped nuScenes category: {category}")

        detections_state.append(bounding_box.array)
        detections_velocity.append(velocity_3d)
        detections_token.append(ann_token)
        detections_types.append(det_type)

    return detections_state, detections_velocity, detections_token, detections_types


def _extract_ego_state(
        nusc: NuScenes,
        sample: Dict[str, Any],
        can_bus: NuScenesCanBus
) -> List[float]:
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    yaw, pitch, roll = Quaternion(ego_pose["rotation"]).yaw_pitch_roll

    # TODO get_nuscenes_pacifica_parameters()
    vehicle_parameters = get_nuplan_pacifica_parameters()

    pose = StateSE3(
        x=ego_pose["translation"][0],
        y=ego_pose["translation"][1],
        z=ego_pose["translation"][2],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )

    #Attempt to obtain the speed from the CAN bus 
    try:
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        pose_msgs = can_bus.get_messages(scene_name, "pose")
        if pose_msgs:
            pose_msg = min(pose_msgs, key=lambda x: abs(x["utime"] - sample["timestamp"]))
            velocity = [pose_msg["vel"][0], pose_msg["vel"][1], pose_msg["vel"][2]]
            acceleration = [pose_msg["accel"][0], pose_msg["accel"][1], pose_msg["accel"][2]]
            angular_velocity = [pose_msg["rotation_rate"][0], pose_msg["rotation_rate"][1],
                                pose_msg["rotation_rate"][2]]
        else:
            velocity = [0.0, 0.0, 0.0]
            acceleration = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
    except:
        velocity = [0.0, 0.0, 0.0]
        acceleration = [0.0, 0.0, 0.0]
        angular_velocity = [0.0, 0.0, 0.0]

    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(*velocity),
        acceleration=Vector3D(*acceleration),
        angular_velocity=Vector3D(*angular_velocity),
    )

    return EgoStateSE3(
        center_se3=pose,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,
    ).array.tolist()