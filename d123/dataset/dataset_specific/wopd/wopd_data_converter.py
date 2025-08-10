import gc
import hashlib
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import tensorflow as tf
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map
from pyquaternion import Quaternion
from waymo_open_dataset import dataset_pb2

from d123.common.datatypes.detection.detection import TrafficLightStatus
from d123.common.datatypes.detection.detection_types import DetectionType
from d123.common.datatypes.sensor.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.common.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3, EgoStateSE3Index
from d123.common.datatypes.vehicle_state.vehicle_parameters import get_wopd_pacifica_parameters
from d123.common.geometry.base import Point3D, StateSE3
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3Index
from d123.common.geometry.constants import DEFAULT_PITCH, DEFAULT_ROLL
from d123.common.geometry.transform.se3 import convert_relative_to_absolute_se3_array, get_rotation_matrix
from d123.common.geometry.vector import Vector3D, Vector3DIndex
from d123.dataset.arrow.helper import open_arrow_table, write_arrow_table
from d123.dataset.dataset_specific.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.dataset.dataset_specific.wopd.waymo_map_utils.wopd_map_utils import convert_wopd_map
from d123.dataset.dataset_specific.wopd.wopd_utils import parse_range_image_and_camera_projection
from d123.dataset.logs.log_metadata import LogMetadata

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
D123_MAPS_ROOT = Path(os.environ.get("D123_MAPS_ROOT"))

TARGET_DT: Final[float] = 0.1
SORT_BY_TIMESTAMP: Final[bool] = True

NUPLAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "red": TrafficLightStatus.RED,
    "unknown": TrafficLightStatus.UNKNOWN,
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto#L63
WOPD_DETECTION_NAME_DICT = {
    0: DetectionType.GENERIC_OBJECT,  # TYPE_UNKNOWN
    1: DetectionType.VEHICLE,  # TYPE_VEHICLE
    2: DetectionType.PEDESTRIAN,  # TYPE_PEDESTRIAN
    3: DetectionType.SIGN,  # TYPE_SIGN
    4: DetectionType.BICYCLE,  # TYPE_CYCLIST
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L50
WOPD_CAMERA_TYPES = {
    1: CameraType.CAM_F0,  # front_camera
    2: CameraType.CAM_L0,  # front_left_camera
    3: CameraType.CAM_R0,  # front_right_camera
    4: CameraType.CAM_L1,  # left_camera
    5: CameraType.CAM_R1,  # right_camera
}

WOPD_DATA_ROOT = Path("/media/nvme1/waymo_perception")  # TODO: set as environment variable

# Whether to use ego or zero roll and pitch values for bounding box detections (after global conversion)
DETECTION_ROLL_PITCH: Final[Literal["ego", "zero"]] = "zero"


def create_token(input_data: str) -> str:
    # TODO: Refactor this function.
    # TODO: Add a general function to create tokens from arbitrary data.
    if isinstance(input_data, str):
        input_data = input_data.encode("utf-8")

    hash_obj = hashlib.sha256(input_data)
    return hash_obj.hexdigest()[:16]


class WOPDDataConverter(RawDataConverter):
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
        self._tf_records_per_split: Dict[str, List[Path]] = self._collect_tf_records()
        self._target_dt: float = 0.1

    def _collect_tf_records(self) -> Dict[str, List[Path]]:
        tf_records_per_split: Dict[str, List[Path]] = {}

        for split in self._splits:
            if split in ["wopd_train"]:
                log_path = WOPD_DATA_ROOT / "training"
            else:
                raise ValueError(f"Split {split} is not supported.")

            log_paths = [log_file for log_file in log_path.glob("*.tfrecord")]
            tf_records_per_split[split] = log_paths

        return tf_records_per_split

    def get_available_splits(self) -> List[str]:
        # TODO: Add more splits if available
        return [
            "wopd_train",
        ]

    def convert_maps(self, worker: WorkerPool) -> None:
        log_args = [
            {
                "tf_record": tf_record,
                "split": split,
            }
            for split, tf_record_paths in self._tf_records_per_split.items()
            for tf_record in tf_record_paths
        ]

        worker_map(
            worker,
            partial(convert_wopd_tfrecord_map_to_gpkg, data_converter_config=self.data_converter_config),
            log_args,
        )

    def convert_logs(self, worker: WorkerPool) -> None:
        log_args = [
            {
                "tf_record": tf_record,
                "split": split,
            }
            for split, tf_record_paths in self._tf_records_per_split.items()
            for tf_record in tf_record_paths
        ]

        worker_map(
            worker,
            partial(convert_wopd_tfrecord_log_to_arrow, data_converter_config=self.data_converter_config),
            log_args,
        )


def convert_wopd_tfrecord_map_to_gpkg(
    args: List[Dict[str, Union[List[str], List[Path]]]], data_converter_config: DataConverterConfig
) -> List[Any]:

    for log_info in args:
        tf_record_path: Path = log_info["tf_record"]
        split: str = log_info["split"]

        if not tf_record_path.exists():
            raise FileNotFoundError(f"TFRecord path {tf_record_path} does not exist.")

        dataset = tf.data.TFRecordDataset(tf_record_path, compression_type="")
        for data in dataset:
            initial_frame = dataset_pb2.Frame()
            initial_frame.ParseFromString(data.numpy())
            break
        log_name = str(initial_frame.context.name)
        map_file_path = D123_MAPS_ROOT / split / f"{log_name}.gpkg"

        if data_converter_config.force_map_conversion or not map_file_path.exists():
            map_file_path.unlink(missing_ok=True)
            convert_wopd_map(initial_frame, map_file_path)
    return []


def convert_wopd_tfrecord_log_to_arrow(
    args: List[Dict[str, Union[List[str], List[Path]]]], data_converter_config: DataConverterConfig
) -> List[Any]:
    for log_info in args:
        try:

            tf_record_path: Path = log_info["tf_record"]
            split: str = log_info["split"]

            if not tf_record_path.exists():
                raise FileNotFoundError(f"TFRecord path {tf_record_path} does not exist.")

            dataset = tf.data.TFRecordDataset(tf_record_path, compression_type="")
            for data in dataset:
                initial_frame = dataset_pb2.Frame()
                initial_frame.ParseFromString(data.numpy())
                break

            log_name = str(initial_frame.context.name)
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
                        raise NotImplementedError("Filepath lidar storage is not implemented.")
                    elif data_converter_config.lidar_store_option == "binary":
                        schema_column_list.append(("lidar", pa.list_(pa.list_(pa.float32(), 6))))

                # TODO: Adjust how cameras are added
                if data_converter_config.camera_store_option is not None:
                    for camera_type in WOPD_CAMERA_TYPES.values():
                        if data_converter_config.camera_store_option == "path":
                            raise NotImplementedError("Path camera storage is not implemented.")
                        elif data_converter_config.camera_store_option == "binary":
                            schema_column_list.append((camera_type.serialize(), pa.binary()))
                            schema_column_list.append(
                                (f"{camera_type.serialize()}_extrinsic", pa.list_(pa.float64(), 4 * 4))
                            )

                recording_schema = pa.schema(schema_column_list)
                metadata = LogMetadata(
                    dataset="wopd",
                    log_name=log_name,
                    location=None,  # TODO: implement map name
                    timestep_seconds=TARGET_DT,  # TODO: Check if correct. Maybe not hardcode
                    map_has_z=True,
                )
                vehicle_parameters = get_wopd_pacifica_parameters()
                camera_metadata = get_wopd_camera_metadata(initial_frame)
                recording_schema = recording_schema.with_metadata(
                    {
                        "log_metadata": json.dumps(asdict(metadata)),
                        "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                        "camera_metadata": camera_metadata_dict_to_json(camera_metadata),
                    }
                )

                _write_recording_table(dataset, recording_schema, log_file_path, tf_record_path, data_converter_config)

                del recording_schema, vehicle_parameters, dataset
        except Exception as e:
            import traceback

            print(f"Error processing log {str(tf_record_path)}: {e}")
            traceback.print_exc()
        gc.collect()
    return []


def get_wopd_camera_metadata(initial_frame: dataset_pb2.Frame) -> Dict[str, CameraMetadata]:

    cam_metadatas: Dict[str, CameraMetadata] = {}
    for calibration in initial_frame.context.camera_calibrations:
        camera_type = WOPD_CAMERA_TYPES[calibration.name]

        # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L96
        # https://github.com/waymo-research/waymo-open-dataset/issues/834#issuecomment-2134995440
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = calibration.intrinsic
        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array([k1, k2, p1, p2, k3])

        if camera_type in WOPD_CAMERA_TYPES.values():
            cam_metadatas[camera_type.serialize()] = CameraMetadata(
                camera_type=camera_type,
                width=calibration.width,
                height=calibration.height,
                intrinsic=_intrinsics,
                distortion=_distortions,
            )

    return cam_metadatas


def _write_recording_table(
    dataset: tf.data.TFRecordDataset,
    recording_schema: pa.schema,
    log_file_path: Path,
    tf_record_path: Path,
    data_converter_config: DataConverterConfig,
) -> None:

    # with pa.ipc.new_stream(str(log_file_path), recording_schema) as writer:
    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:

            dataset = tf.data.TFRecordDataset(tf_record_path, compression_type="")
            for frame_idx, data in enumerate(dataset):
                frame = dataset_pb2.Frame()
                frame.ParseFromString(data.numpy())

                (detections_state, detections_velocity, detections_token, detections_types) = _extract_detections(frame)
                # traffic_light_ids, traffic_light_types = _extract_traffic_lights(log_db, lidar_pc_token)
                # route_lane_group_ids = [
                #     int(roadblock_id)
                #     for roadblock_id in str(lidar_pc.scene.roadblock_ids).split(" ")
                #     if len(roadblock_id) > 0
                # ]

                # TODO: Implement traffic light extraction
                traffic_light_ids = []
                traffic_light_types = []

                # TODO: Implement detections
                row_data = {
                    "token": [create_token(f"{frame.context.name}_{int(frame.timestamp_micros)}")],
                    "timestamp": [int(frame.timestamp_micros)],
                    "detections_state": [detections_state],
                    "detections_velocity": [detections_velocity],
                    "detections_token": [detections_token],
                    "detections_type": [detections_types],
                    "ego_states": [_extract_ego_state(frame)],
                    "traffic_light_ids": [traffic_light_ids],
                    "traffic_light_types": [traffic_light_types],
                    "scenario_tag": ["unknown"],
                    "route_lane_group_ids": [None],
                }

                # TODO: Implement lidar extraction
                if data_converter_config.lidar_store_option is not None:
                    row_data["lidar"] = [_extract_lidar(frame, data_converter_config).tolist()]

                if data_converter_config.camera_store_option is not None:
                    camera_data_dict = _extract_camera(frame, data_converter_config)
                    for camera_type, camera_data in camera_data_dict.items():
                        if camera_data is not None:
                            row_data[camera_type.serialize()] = [camera_data[0]]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [camera_data[1]]
                        else:
                            row_data[camera_type.serialize()] = [None]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [None]

                batch = pa.record_batch(row_data, schema=recording_schema)
                writer.write_batch(batch)
                del batch, row_data, detections_state, detections_velocity, detections_token, detections_types

    if SORT_BY_TIMESTAMP:
        recording_table = open_arrow_table(log_file_path)
        recording_table = recording_table.sort_by([("timestamp", "ascending")])
        write_arrow_table(recording_table, log_file_path)


# def _get_ego_pose_se3(frame: dataset_pb2.Frame) -> StateSE3:
#     ego_pose_matrix = np.array(frame.pose.transform).reshape(4, 4)
#     yaw, pitch, roll = Quaternion(matrix=ego_pose_matrix[:3, :3]).yaw_pitch_roll
#     ego_point_3d = Point3D.from_array(ego_pose_matrix[:3, 3])

#     # TODO: figure out if ego frame is given in rear axle or center frame
#     return StateSE3(x=ego_point_3d.x, y=ego_point_3d.y, z=ego_point_3d.z, roll=pitch, pitch=-roll, yaw=yaw)


def _get_ego_pose_se3(frame: dataset_pb2.Frame) -> StateSE3:
    ego_pose_matrix = np.array(frame.pose.transform).reshape(4, 4)
    yaw, pitch, roll = Quaternion(matrix=ego_pose_matrix[:3, :3]).yaw_pitch_roll
    ego_point_3d = Point3D.from_array(ego_pose_matrix[:3, 3])

    return StateSE3(x=ego_point_3d.x, y=ego_point_3d.y, z=ego_point_3d.z, roll=roll, pitch=pitch, yaw=yaw)

    # TODO: figure out if ego frame is given in rear axle or center frame
    # return StateSE3(x=ego_point_3d.x, y=ego_point_3d.y, z=ego_point_3d.z, roll=pitch, pitch=-roll, yaw=yaw)


def _extract_detections(frame: dataset_pb2.Frame) -> Tuple[List[List[float]], List[List[float]], List[str], List[int]]:
    # TODO: implement

    ego_rear_axle = _get_ego_pose_se3(frame)

    num_detections = len(frame.laser_labels)
    detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    detections_token: List[str] = []
    detections_types: List[int] = []

    for detection_idx, detection in enumerate(frame.laser_labels):
        if detection.type not in WOPD_DETECTION_NAME_DICT:
            continue

        # 1. SS3 Bounding Box
        detections_state[detection_idx, BoundingBoxSE3Index.X] = detection.box.center_x
        detections_state[detection_idx, BoundingBoxSE3Index.Y] = detection.box.center_y
        detections_state[detection_idx, BoundingBoxSE3Index.Z] = detection.box.center_z
        detections_state[detection_idx, BoundingBoxSE3Index.ROLL] = DEFAULT_ROLL  # not provided in WOPD
        detections_state[detection_idx, BoundingBoxSE3Index.PITCH] = DEFAULT_PITCH  # not provided in WOPD
        detections_state[detection_idx, BoundingBoxSE3Index.YAW] = detection.box.heading
        detections_state[detection_idx, BoundingBoxSE3Index.LENGTH] = detection.box.length
        detections_state[detection_idx, BoundingBoxSE3Index.WIDTH] = detection.box.width
        detections_state[detection_idx, BoundingBoxSE3Index.HEIGHT] = detection.box.height

        # 2. Velocity TODO: check if velocity needs to be rotated
        detections_velocity[detection_idx] = Vector3D(
            x=detection.metadata.speed_x,
            y=detection.metadata.speed_y,
            z=detection.metadata.speed_z,
        ).array

        # 3. Type and track token
        detections_token.append(str(detection.id))
        detections_types.append(int(WOPD_DETECTION_NAME_DICT[detection.type]))

    detections_state[:, BoundingBoxSE3Index.STATE_SE3] = convert_relative_to_absolute_se3_array(
        origin=ego_rear_axle, se3_array=detections_state[:, BoundingBoxSE3Index.STATE_SE3]
    )
    if DETECTION_ROLL_PITCH == "ego":
        pass
    if DETECTION_ROLL_PITCH == "zero":
        detections_state[:, BoundingBoxSE3Index.ROLL] = DEFAULT_ROLL
        detections_state[:, BoundingBoxSE3Index.PITCH] = DEFAULT_PITCH
    else:
        raise ValueError(f"Invalid DETECTION_ROLL_PITCH value: {DETECTION_ROLL_PITCH}. Must be 'ego' or 'zero'.")

    return detections_state.tolist(), detections_velocity.tolist(), detections_token, detections_types


def _extract_ego_state(frame: dataset_pb2.Frame) -> List[float]:
    rear_axle_pose = _get_ego_pose_se3(frame)

    vehicle_parameters = get_wopd_pacifica_parameters()
    # FIXME: Find dynamic state in waymo open perception dataset
    # https://github.com/waymo-research/waymo-open-dataset/issues/55#issuecomment-546152290
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(*np.zeros(3)),
        acceleration=Vector3D(*np.zeros(3)),
        angular_velocity=Vector3D(*np.zeros(3)),
    )

    return EgoStateSE3.from_rear_axle(
        rear_axle_se3=rear_axle_pose,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        time_point=None,
    ).array.tolist()


def _extract_traffic_lights() -> Tuple[List[int], List[int]]:
    pass


def _extract_camera(
    frame: dataset_pb2.Frame, data_converter_config: DataConverterConfig
) -> Dict[CameraType, Union[str, bytes]]:

    camera_dict: Dict[str, Union[str, bytes]] = {}  # TODO: Fix wrong type hint
    np.array(frame.pose.transform).reshape(4, 4)

    # NOTE: The extrinsic matrix in frame.context.camera_calibration is fixed to model the ego to camera transformation.
    # The poses in frame.images[idx] are the motion compensated ego poses when the camera triggers.
    #

    context_extrinsic: Dict[str, npt.NDArray] = {}
    for calibration in frame.context.camera_calibrations:
        camera_type = WOPD_CAMERA_TYPES[calibration.name]

        transform = np.array(calibration.extrinsic.transform).reshape(4, 4)

        # FIXME: This is an ugly hack to convert to uniform camera convention.
        flip_camera = get_rotation_matrix(
            StateSE3(
                x=0.0,
                y=0.0,
                z=0.0,
                roll=np.deg2rad(0.0),
                pitch=np.deg2rad(90.0),
                yaw=np.deg2rad(-90.0),
            )
        )
        transform[:3, :3] = transform[:3, :3] @ flip_camera
        context_extrinsic[camera_type] = transform

    for image_proto in frame.images:
        camera_type = WOPD_CAMERA_TYPES[image_proto.name]

        np.array(image_proto.pose.transform).reshape(4, 4)
        camera_bytes = image_proto.image

        # # Compute the transform from ego_global_transform to ego_at_camera_transform
        # # ego_global_transform * T = ego_at_camera_transform  =>  T = ego_global_transform^-1 * ego_at_camera_transform
        # np.linalg.inv(ego_global_transform) @ ego_at_trigger_transform

        # TODO: figure out the correct transform
        camera_dict[camera_type] = camera_bytes, context_extrinsic[camera_type].flatten().tolist()

    return camera_dict


def _extract_lidar(
    frame: dataset_pb2.Frame, data_converter_config: DataConverterConfig
) -> Optional[npt.NDArray[np.float32]]:
    from waymo_open_dataset.utils import frame_utils

    assert data_converter_config.lidar_store_option == "binary", "Lidar store option must be 'binary' for WOPD."
    (range_images, camera_projections, _, range_image_top_pose) = parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame=frame,
        range_images=range_images,
        camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose,
        keep_polar_features=True,
    )
    points = np.array(points[0], dtype=np.float32)
    return points
