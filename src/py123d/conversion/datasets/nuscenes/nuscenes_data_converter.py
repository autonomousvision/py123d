import gc
import json
import os
import numpy as np
import pyarrow as pa

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from py123d.conversion.log_writer.abstract_log_writer import AbstractLogWriter, LiDARData
from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.script.builders.worker_pool_builder import WorkerPool
from py123d.datatypes.detections.box_detections import (
    BoxDetectionSE3,
    BoxDetectionWrapper,
    BoxDetectionMetadata,)
from py123d.datatypes.detections.traffic_light_detections import (TrafficLightDetection,
    TrafficLightDetectionWrapper,
)
from py123d.datatypes.detections.box_detection_types import BoxDetectionType
from py123d.datatypes.maps.map_metadata import MapMetadata
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from py123d.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
from py123d.datatypes.sensors.lidar.lidar_index import NuscenesLidarIndex

from py123d.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3, EgoStateSE3Index
from py123d.datatypes.vehicle_state.vehicle_parameters import get_nuscenes_renauly_zoe_parameters
from py123d.geometry import StateSE3, BoundingBoxSE3, BoundingBoxSE3Index
from py123d.geometry.vector import Vector3D, Vector3DIndex
from py123d.common.utils.arrow_helper import open_arrow_table, write_arrow_table
from py123d.conversion.datasets.nuscenes.nuscenes_map_conversion import write_nuscenes_map, NUSCENES_MAPS
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.datatypes.time.time_point import TimePoint

TARGET_DT: Final[float] = 0.1
NUSCENES_DT: Final[float] = 0.5
SORT_BY_TIMESTAMP: Final[bool] = True
NUSCENES_DETECTION_NAME_DICT = {
    # Vehicles (4+ wheels)
    "vehicle.car": BoxDetectionType.VEHICLE,
    "vehicle.truck": BoxDetectionType.VEHICLE,
    "vehicle.bus.bendy": BoxDetectionType.VEHICLE,
    "vehicle.bus.rigid": BoxDetectionType.VEHICLE,
    "vehicle.construction": BoxDetectionType.VEHICLE,
    "vehicle.emergency.ambulance": BoxDetectionType.VEHICLE,
    "vehicle.emergency.police": BoxDetectionType.VEHICLE,
    "vehicle.trailer": BoxDetectionType.VEHICLE,

    # Bicycles / Motorcycles
    "vehicle.bicycle": BoxDetectionType.BICYCLE,
    "vehicle.motorcycle": BoxDetectionType.BICYCLE,

    # Pedestrians (all subtypes)
    "human.pedestrian.adult": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.child": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.construction_worker": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.personal_mobility": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.police_officer": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.stroller": BoxDetectionType.PEDESTRIAN,
    "human.pedestrian.wheelchair": BoxDetectionType.PEDESTRIAN,

    # Traffic cone / barrier
    "movable_object.trafficcone": BoxDetectionType.TRAFFIC_CONE,
    "movable_object.barrier": BoxDetectionType.BARRIER,

    # Generic objects
    "movable_object.pushable_pullable": BoxDetectionType.GENERIC_OBJECT,
    "movable_object.debris": BoxDetectionType.GENERIC_OBJECT,
    "static_object.bicycle_rack": BoxDetectionType.GENERIC_OBJECT,
    "animal": BoxDetectionType.GENERIC_OBJECT,
}

NUSCENES_CAMERA_TYPES = {
    PinholeCameraType.CAM_F0: "CAM_FRONT",
    PinholeCameraType.CAM_B0: "CAM_BACK",
    PinholeCameraType.CAM_L0: "CAM_FRONT_LEFT",
    PinholeCameraType.CAM_L1: "CAM_BACK_LEFT",
    PinholeCameraType.CAM_R0: "CAM_FRONT_RIGHT",
    PinholeCameraType.CAM_R1: "CAM_BACK_RIGHT",
}
NUSCENES_DATA_ROOT = Path(os.environ["NUSCENES_DATA_ROOT"])


class NuScenesDataConverter(AbstractDatasetConverter):
    def __init__(
            self,
            splits: List[str],
            nuscenes_data_root: Union[Path, str],
            nuscenes_lanelet2_root: Union[Path, str],
            use_lanelet2: bool,
            dataset_converter_config: DatasetConverterConfig,
            version: str = "v1.0-trainval",
    ) -> None:
        super().__init__(dataset_converter_config)
        self._splits: List[str] = splits
        self._nuscenes_data_root: Path = Path(nuscenes_data_root)
        self._nuscenes_lanelet2_root: Path = Path(nuscenes_lanelet2_root)
        self._use_lanelet2 = use_lanelet2
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

            # get token
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

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(NUSCENES_MAPS)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return sum(len(scene_tokens) for scene_tokens in self._scene_tokens_per_split.values())

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        map_name = NUSCENES_MAPS[map_index]

        map_metadata = _get_nuscenes_map_metadata(map_name)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)

        if map_needs_writing:
            write_nuscenes_map(
                nuscenes_maps_root=self._nuscenes_data_root,
                location=map_name,
                map_writer=map_writer,
                use_lanelet2=self._use_lanelet2,
                lanelet2_root=Path(self._nuscenes_lanelet2_root),
            )

        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""
        # Find the scene token for the given log index
        all_scene_tokens = []
        for split, scene_tokens in self._scene_tokens_per_split.items():
            all_scene_tokens.extend([(split, token) for token in scene_tokens])

        if log_index >= len(all_scene_tokens):
            raise ValueError(f"Log index {log_index} is out of range. Total logs: {len(all_scene_tokens)}")

        split, scene_token = all_scene_tokens[log_index]

        nusc = NuScenes(version=self._version, dataroot=str(self._nuscenes_data_root), verbose=False)
        scene = nusc.get("scene", scene_token)
        log_record = nusc.get("log", scene["log_token"])

        # 1. Initialize log metadata
        log_metadata = LogMetadata(
            dataset="nuscenes",
            split=split,
            log_name=scene["name"],
            location=log_record["location"],
            timestep_seconds=TARGET_DT,
            vehicle_parameters=get_nuscenes_renauly_zoe_parameters(),
            camera_metadata=_get_nuscenes_camera_metadata(nusc, scene, self.dataset_converter_config),
            lidar_metadata=_get_nuscenes_lidar_metadata(nusc, scene, self.dataset_converter_config),
            map_metadata=_get_nuscenes_map_metadata(log_record["location"]),
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        if log_needs_writing:
            can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))

            step_interval = max(1, int(TARGET_DT / NUSCENES_DT))
            sample_count = 0

            # Traverse all samples in the scene
            sample_token = scene["first_sample_token"]
            while sample_token:
                if sample_count % step_interval == 0:
                    sample = nusc.get("sample", sample_token)

                    log_writer.write(
                        timestamp=TimePoint.from_us(sample["timestamp"]),
                        ego_state=_extract_nuscenes_ego_state(nusc, sample, can_bus),
                        box_detections=_extract_nuscenes_box_detections(nusc, sample),
                        traffic_lights=_extract_nuscenes_traffic_lights(),  # nuScenes doesn't have traffic lights
                        cameras=_extract_nuscenes_cameras(
                            nusc=nusc,
                            sample=sample,
                            dataset_converter_config=self.dataset_converter_config,
                        ),
                        lidars=_extract_nuscenes_lidars(
                            nusc=nusc,
                            sample=sample,
                            dataset_converter_config=self.dataset_converter_config,
                        ),
                        scenario_tags=_extract_nuscenes_scenario_tag(),  # nuScenes doesn't have scenario tags
                        route_lane_group_ids=_extract_nuscenes_route_lane_group_ids(),
                        # nuScenes doesn't have route info
                    )

                sample_token = sample["next"]
                sample_count += 1

        log_writer.close()
        del nusc
        gc.collect()

    def convert_logs(self, worker: WorkerPool) -> None:
        """
        NuScenes logs conversion is handled externally through convert_log method.
        This method is kept for interface compatibility.
        """
        pass


def _get_nuscenes_camera_metadata(
        nusc: NuScenes,
        scene: Dict[str, Any],
        dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, PinholeCameraMetadata]:
    camera_metadata: Dict[PinholeCameraType, PinholeCameraMetadata] = {}

    if dataset_converter_config.include_cameras:
        first_sample_token = scene["first_sample_token"]
        first_sample = nusc.get("sample", first_sample_token)

        for camera_type, camera_channel in NUSCENES_CAMERA_TYPES.items():
            cam_token = first_sample["data"][camera_channel]
            cam_data = nusc.get("sample_data", cam_token)
            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

            intrinsic_matrix = np.array(calib["camera_intrinsic"])
            intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsic_matrix)
            distortion = PinholeDistortion.from_array(np.zeros(5), copy=False)

            camera_metadata[camera_type] = PinholeCameraMetadata(
                camera_type=camera_type,
                width=cam_data["width"],
                height=cam_data["height"],
                intrinsics=intrinsic,
                distortion=distortion,
            )

    return camera_metadata


def _get_nuscenes_lidar_metadata(
        nusc: NuScenes,
        scene: Dict[str, Any],
        dataset_converter_config: DatasetConverterConfig,
) -> Dict[LiDARType, LiDARMetadata]:
    metadata: Dict[LiDARType, LiDARMetadata] = {}

    if dataset_converter_config.include_lidars:
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
        extrinsic = StateSE3.from_transformation_matrix(extrinsic)

        metadata[LiDARType.LIDAR_MERGED] = LiDARMetadata(
            lidar_type=LiDARType.LIDAR_MERGED,
            lidar_index=NuscenesLidarIndex,
            extrinsic=extrinsic,
        )

    return metadata


def _get_nuscenes_map_metadata(location):
    return MapMetadata(
        dataset="nuscenes",
        split=None,
        log_name=None,
        location=location,
        map_has_z=False,
        map_is_local=False,
    )


def _extract_nuscenes_ego_state(nusc, sample, can_bus) -> EgoStateSE3:
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    quat = Quaternion(ego_pose["rotation"])

    vehicle_parameters = get_nuscenes_renauly_zoe_parameters()
    pose = StateSE3(
        x=ego_pose["translation"][0],
        y=ego_pose["translation"][1],
        z=ego_pose["translation"][2],
        qw=quat.w,
        qx=quat.x,
        qy=quat.y,
        qz=quat.z,
    )

    scene_name = nusc.get("scene", sample["scene_token"])["name"]

    try:
        pose_msgs = can_bus.get_messages(scene_name, "pose")
    except Exception as e:
        pose_msgs = []

    if pose_msgs:
        closest_msg = None
        min_time_diff = float('inf')
        for msg in pose_msgs:
            time_diff = abs(msg["utime"] - sample["timestamp"])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_msg = msg

        if closest_msg and min_time_diff < 500000:
            velocity = [*closest_msg["vel"]]
            acceleration = [*closest_msg["accel"]]
            angular_velocity = [*closest_msg["rotation_rate"]]
        else:
            velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]
    else:
        velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]

    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(*velocity),
        acceleration=Vector3D(*acceleration),
        angular_velocity=Vector3D(*angular_velocity),
    )

    return EgoStateSE3(
        center_se3=pose,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=TimePoint.from_us(sample["timestamp"]),
    )


def _extract_nuscenes_box_detections(
        nusc: NuScenes,
        sample: Dict[str, Any]
) -> BoxDetectionWrapper:
    box_detections: List[BoxDetectionSE3] = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

        box_quat = box.orientation
        euler_angles = box_quat.yaw_pitch_roll  # (yaw, pitch, roll)

        # Create StateSE3 for box center and orientation
        center_quat = box.orientation
        center = StateSE3(
            box.center[0],
            box.center[1],
            box.center[2],
            center_quat.w,
            center_quat.x,
            center_quat.y,
            center_quat.z,
        )
        bounding_box = BoundingBoxSE3(center, box.wlh[1], box.wlh[0], box.wlh[2])
        # Get detection type
        category = ann["category_name"]
        det_type = None
        for key, value in NUSCENES_DETECTION_NAME_DICT.items():
            if category.startswith(key):
                det_type = value
                break

        if det_type is None:
            print(f"Warning: Unmapped nuScenes category: {category}, skipping")
            continue

        # Get velocity if available
        velocity = nusc.box_velocity(ann_token)
        velocity_3d = Vector3D(x=velocity[0], y=velocity[1], z=velocity[2] if len(velocity) > 2 else 0.0)

        metadata = BoxDetectionMetadata(
            box_detection_type=det_type,
            track_token=ann["instance_token"],
            timepoint=TimePoint.from_us(sample["timestamp"]),
            confidence=1.0,  # nuScenes annotations are ground truth
            num_lidar_points=ann.get("num_lidar_pts", 0),
        )

        box_detection = BoxDetectionSE3(
            metadata=metadata,
            bounding_box_se3=bounding_box,
            velocity=velocity_3d,
        )
        box_detections.append(box_detection)

    return BoxDetectionWrapper(box_detections=box_detections)


def _extract_nuscenes_traffic_lights() -> TrafficLightDetectionWrapper:
    """nuScenes doesn't have traffic light information."""
    return TrafficLightDetectionWrapper(traffic_light_detections=[])


def _extract_nuscenes_cameras(
    nusc: NuScenes,
    sample: Dict[str, Any],
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]]:
    camera_dict: Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]] = {}

    if dataset_converter_config.include_cameras:
        for camera_type, camera_channel in NUSCENES_CAMERA_TYPES.items():
            cam_token = sample["data"][camera_channel]
            cam_data = nusc.get("sample_data", cam_token)

            # Check timestamp synchronization (within 100ms)
            if abs(cam_data["timestamp"] - sample["timestamp"]) > 100000:
                continue

            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

            translation = np.array(calib["translation"])
            rotation = Quaternion(calib["rotation"]).rotation_matrix
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = rotation
            extrinsic_matrix[:3, 3] = translation
            extrinsic = StateSE3.from_transformation_matrix(extrinsic_matrix)

            cam_path = NUSCENES_DATA_ROOT / cam_data["filename"]

            if cam_path.exists() and cam_path.is_file():
                if dataset_converter_config.camera_store_option == "path":
                    # TODO: should be relative path
                    camera_data = cam_data["filename"]
                elif dataset_converter_config.camera_store_option == "binary":
                    with open(cam_path, "rb") as f:
                        camera_data = f.read()
                else:
                    continue

                camera_dict[camera_type] = (camera_data, extrinsic)

    return camera_dict


def _extract_nuscenes_lidars(
        nusc: NuScenes,
        sample: Dict[str, Any],
        dataset_converter_config: DatasetConverterConfig,
) -> List[LiDARData]:
    lidars: List[LiDARData] = []

    if dataset_converter_config.include_lidars:
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)
        lidar_path = NUSCENES_DATA_ROOT / lidar_data["filename"]

        if lidar_path.exists() and lidar_path.is_file():
            lidar = LiDARData(
                lidar_type=LiDARType.LIDAR_TOP,
                relative_path=str(lidar_path),
                dataset_root=NUSCENES_DATA_ROOT, 
                iteration=lidar_data.get("iteration"),
            )
            lidars.append(lidar)
        else:
            lidars.append(LiDARData(
                lidar_type=LiDARType.LIDAR_TOP,
                relative_path=None,
                dataset_root=NUSCENES_DATA_ROOT,
            ))

    return lidars

def _extract_nuscenes_scenario_tag() -> List[str]:
    """nuScenes doesn't have scenario tags."""
    return ["unknown"]


def _extract_nuscenes_route_lane_group_ids() -> List[int]:
    """nuScenes doesn't have route lane group information."""
    return []


# Updated arrow conversion function using the new extraction functions
def convert_nuscenes_log_to_arrow(
    args: List[Dict[str, Union[str, List[str]]]],
    dataset_converter_config: DatasetConverterConfig,
    version: str
) -> List[Any]:
    for log_info in args:
        scene_token: str = log_info["scene_token"]
        split: str = log_info["split"]

        nusc = NuScenes(version=version, dataroot=str(NUSCENES_DATA_ROOT), verbose=False)
        scene = nusc.get("scene", scene_token)

        log_file_path = dataset_converter_config.output_path / split / f"{scene_token}.arrow"

        if dataset_converter_config.force_log_conversion or not log_file_path.exists():
            if not log_file_path.parent.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Define schema
            schema_column_list = [
                ("token", pa.string()),
                ("timestamp", pa.int64()),
                ("ego_state", pa.list_(pa.float64(), len(EgoStateSE3Index))),
                ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                ("detections_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                ("detections_token", pa.list_(pa.string())),
                ("detections_type", pa.list_(pa.int16())),
                ("traffic_light_ids", pa.list_(pa.int64())),
                ("traffic_light_types", pa.list_(pa.int16())),
                ("scenario_tag", pa.list_(pa.string())),
                ("route_lane_group_ids", pa.list_(pa.int64())),
            ]

            if dataset_converter_config.lidar_store_option == "path":
                schema_column_list.append(("lidar", pa.string()))

            if dataset_converter_config.camera_store_option == "path":
                for camera_type in NUSCENES_CAMERA_TYPES.keys():
                    schema_column_list.append((camera_type.serialize(), pa.string()))
                    schema_column_list.append((f"{camera_type.serialize()}_extrinsic", pa.list_(pa.float64(), 4 * 4)))

            recording_schema = pa.schema(schema_column_list)

            log_record = nusc.get("log", scene["log_token"])
            location = log_record["location"]

            # Create metadata using the same functions as the new interface
            metadata = LogMetadata(
                dataset="nuscenes",
                split=split,
                log_name=scene["name"],
                location=location,
                timestep_seconds=TARGET_DT,
                vehicle_parameters=get_nuscenes_renauly_zoe_parameters(),
                camera_metadata=_get_nuscenes_camera_metadata(nusc, scene, dataset_converter_config),
                lidar_metadata=_get_nuscenes_lidar_metadata(nusc, scene, dataset_converter_config),
                map_metadata=_get_nuscenes_map_metadata(location),
            )

            recording_schema = recording_schema.with_metadata(
                {
                    "log_metadata": json.dumps(asdict(metadata)),
                }
            )

            _write_arrow_table_with_new_interface(
                nusc, scene, recording_schema, log_file_path, dataset_converter_config
            )

        del nusc
        gc.collect()

    return []


def _write_arrow_table_with_new_interface(
    nusc: NuScenes,
    scene: Dict[str, Any],
    recording_schema: pa.schema,
    log_file_path: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> None:
    can_bus = NuScenesCanBus(dataroot=str(NUSCENES_DATA_ROOT))

    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:
            step_interval = max(1, int(TARGET_DT / NUSCENES_DT))
            sample_count = 0

            sample_token = scene["first_sample_token"]
            while sample_token:
                if sample_count % step_interval == 0:
                    sample = nusc.get("sample", sample_token)

                    # Use the new extraction functions for consistency
                    ego_state = _extract_nuscenes_ego_state(nusc, sample, can_bus)
                    box_detections = _extract_nuscenes_box_detections(nusc, sample)
                    cameras = _extract_nuscenes_cameras(nusc, sample, dataset_converter_config)
                    lidars = _extract_nuscenes_lidars(nusc, sample, dataset_converter_config)

                    detections_state_list = []
                    for det in box_detections.box_detections:
                        bbox_array = det.bounding_box_se3.array
                        
                        print(f"bbox_array shape: {bbox_array.shape}, ndim: {bbox_array.ndim}")
                        if bbox_array.ndim > 1:
                            detections_state_list.append(bbox_array.flatten().tolist())
                        else:
                            detections_state_list.append(bbox_array.tolist())

                    # Prepare row data
                    row_data = {
                        "token": [sample_token],
                        "timestamp": [sample["timestamp"]],
                        "ego_state": ego_state.array.flatten().tolist(),
                        "detections_state": detections_state_list,
                        "detections_velocity": [det.velocity.array.tolist() for det in box_detections.box_detections],
                        "detections_token": [det.metadata.track_token for det in box_detections.box_detections],
                        "detections_type": [det.metadata.box_detection_type.value for det in box_detections.box_detections],
                        "traffic_light_ids": [],
                        "traffic_light_types": [],
                        "scenario_tag": ["unknown"],
                        "route_lane_group_ids": [],
                    }

                    # Add lidar data if configured
                    if dataset_converter_config.lidar_store_option == "path":
                        row_data["lidar"] = [lidars.get(LiDARType.LIDAR_MERGED, None)]

                    # Add camera data if configured
                    if dataset_converter_config.camera_store_option == "path":
                        for camera_type in NUSCENES_CAMERA_TYPES.keys():
                            if camera_type in cameras:
                                camera_path, extrinsic = cameras[camera_type]
                                row_data[camera_type.serialize()] = [camera_path]
                                row_data[f"{camera_type.serialize()}_extrinsic"] = [
                                    extrinsic.to_transformation_matrix().flatten().tolist()]
                            else:
                                row_data[camera_type.serialize()] = [None]
                                row_data[f"{camera_type.serialize()}_extrinsic"] = [None]

                    batch = pa.record_batch(row_data, schema=recording_schema)
                    writer.write_batch(batch)

                sample_token = sample["next"]
                sample_count += 1

    # Sort by timestamp if required
    if SORT_BY_TIMESTAMP:
        recording_table = open_arrow_table(log_file_path)
        recording_table = recording_table.sort_by([("timestamp", "ascending")])
        write_arrow_table(recording_table, log_file_path)
