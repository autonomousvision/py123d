import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.wopd.utils.wopd_constants import (
    WOPD_AVAILABLE_SPLITS,
    WOPD_CAMERA_TYPES,
    WOPD_DETECTION_NAME_DICT,
    WOPD_LIDAR_TYPES,
)
from py123d.conversion.datasets.wopd.waymo_map_utils.wopd_map_utils import convert_wopd_map
from py123d.conversion.datasets.wopd.wopd_utils import parse_range_image_and_camera_projection
from py123d.conversion.log_writer.abstract_log_writer import AbstractLogWriter
from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.utils.sensor_utils.camera_conventions import CameraConvention, convert_camera_convention
from py123d.conversion.utils.sensor_utils.lidar_index_registry import DefaultLidarIndex, WOPDLidarIndex
from py123d.datatypes.detections.box_detections import BoxDetectionMetadata, BoxDetectionSE3, BoxDetectionWrapper
from py123d.datatypes.maps.map_metadata import MapMetadata
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from py123d.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import get_wopd_chrysler_pacifica_parameters
from py123d.geometry import (
    BoundingBoxSE3,
    BoundingBoxSE3Index,
    EulerAngles,
    EulerAnglesIndex,
    StateSE3,
    StateSE3Index,
    Vector3D,
    Vector3DIndex,
)
from py123d.geometry.transform.transform_se3 import convert_relative_to_absolute_se3_array
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.geometry.utils.rotation_utils import (
    get_euler_array_from_quaternion_array,
    get_quaternion_array_from_euler_array,
)

check_dependencies(modules=["tensorflow", "waymo_open_dataset"], optional_name="waymo")
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))

logger = logging.getLogger(__name__)


class WOPDConverter(AbstractDatasetConverter):
    def __init__(
        self,
        splits: List[str],
        wopd_data_root: Union[Path, str],
        zero_roll_pitch: bool,
        keep_polar_features: bool,
        add_map_pose_offset: bool,
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        super().__init__(dataset_converter_config)
        for split in splits:
            assert (
                split in WOPD_AVAILABLE_SPLITS
            ), f"Split {split} is not available. Available splits: {WOPD_AVAILABLE_SPLITS}"

        self._splits: List[str] = splits
        self._wopd_data_root: Path = Path(wopd_data_root)
        self._zero_roll_pitch: bool = zero_roll_pitch
        self._keep_polar_features: bool = keep_polar_features
        self._add_map_pose_offset: bool = add_map_pose_offset  # TODO: Implement this feature

        self._split_tf_record_pairs: List[Tuple[str, List[Path]]] = self._collect_split_tf_record_pairs()

    def _collect_split_tf_record_pairs(self) -> Dict[str, List[Path]]:
        """Helper to collect the pairings of the split names and the corresponding tf record file."""

        split_tf_record_pairs: List[Tuple[str, List[Path]]] = []
        split_name_mapping: Dict[str, str] = {
            "wopd_train": "training",
            "wopd_val": "validation",
            "wopd_test": "testing",
        }

        for split in self._splits:
            assert split in split_name_mapping.keys()
            split_folder = self._wopd_data_root / split_name_mapping[split]
            source_log_paths = [log_file for log_file in split_folder.glob("*.tfrecord")]
            for source_log_path in source_log_paths:
                split_tf_record_pairs.append((split, source_log_path))

        return split_tf_record_pairs

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(self._split_tf_record_pairs)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._split_tf_record_pairs)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        split, source_tf_record_path = self._split_tf_record_pairs[map_index]
        initial_frame = _get_initial_frame_from_tfrecord(source_tf_record_path)

        map_metadata = _get_wopd_map_metadata(initial_frame, split)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)
        if map_needs_writing:
            convert_wopd_map(initial_frame, map_writer)

        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        split, source_tf_record_path = self._split_tf_record_pairs[log_index]

        initial_frame = _get_initial_frame_from_tfrecord(source_tf_record_path, keep_dataset=False)
        log_name = str(initial_frame.context.name)
        dataset = tf.data.TFRecordDataset(source_tf_record_path, compression_type="")

        # 1. Initialize Metadata
        log_metadata = LogMetadata(
            dataset="wopd",
            split=split,
            log_name=log_name,
            location=str(initial_frame.context.stats.location),
            timestep_seconds=0.1,
            vehicle_parameters=get_wopd_chrysler_pacifica_parameters(),
            camera_metadata=_get_wopd_camera_metadata(
                initial_frame,
                self.dataset_converter_config,
            ),
            lidar_metadata=_get_wopd_lidar_metadata(
                initial_frame,
                self._keep_polar_features,
                self.dataset_converter_config,
            ),
            map_metadata=_get_wopd_map_metadata(initial_frame, split),
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        # 3. Process source log data
        if log_needs_writing:
            try:
                for frame_idx, data in enumerate(dataset):
                    frame = dataset_pb2.Frame()
                    frame.ParseFromString(data.numpy())

                    map_pose_offset: Vector3D = Vector3D(0.0, 0.0, 0.0)
                    if self._add_map_pose_offset:
                        map_pose_offset = Vector3D(
                            x=frame.map_pose_offset.x,
                            y=frame.map_pose_offset.y,
                            z=frame.map_pose_offset.z,
                        )

                    log_writer.write(
                        timestamp=TimePoint.from_us(frame.timestamp_micros),
                        ego_state=_extract_wopd_ego_state(frame, map_pose_offset),
                        box_detections=_extract_wopd_box_detections(frame, map_pose_offset, self._zero_roll_pitch),
                        traffic_lights=None,  # TODO: Check if WOPD has traffic light information
                        cameras=_extract_wopd_cameras(frame, self.dataset_converter_config),
                        lidars=_extract_wopd_lidars(frame, self._keep_polar_features, self.dataset_converter_config),
                    )
            except Exception as e:
                logger.error(f"Error processing log {log_name}: {e}")

        log_writer.close()


def _get_initial_frame_from_tfrecord(
    tf_record_path: Path,
    keep_dataset: bool = False,
) -> Union[dataset_pb2.Frame, Tuple[dataset_pb2.Frame, tf.data.TFRecordDataset]]:
    dataset = tf.data.TFRecordDataset(tf_record_path, compression_type="")
    for data in dataset:
        initial_frame = dataset_pb2.Frame()
        initial_frame.ParseFromString(data.numpy())
        break

    if keep_dataset:
        return initial_frame, dataset

    del dataset
    return initial_frame


def _get_wopd_map_metadata(initial_frame: dataset_pb2.Frame, split: str) -> MapMetadata:

    map_metadata = MapMetadata(
        dataset="wopd",
        split=split,
        log_name=str(initial_frame.context.name),
        location=None,  # TODO: Add location information.
        map_has_z=True,
        map_is_local=True,  # True, if map is per log
    )

    return map_metadata


def _get_wopd_camera_metadata(
    initial_frame: dataset_pb2.Frame, dataset_converter_config: DatasetConverterConfig
) -> Dict[PinholeCameraType, PinholeCameraMetadata]:

    camera_metadata_dict: Dict[PinholeCameraType, PinholeCameraMetadata] = {}

    if dataset_converter_config.camera_store_option is not None:
        for calibration in initial_frame.context.camera_calibrations:
            camera_type = WOPD_CAMERA_TYPES[calibration.name]
            # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L96
            # https://github.com/waymo-research/waymo-open-dataset/issues/834#issuecomment-2134995440
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = calibration.intrinsic
            intrinsics = PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
            distortion = PinholeDistortion(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
            if camera_type in WOPD_CAMERA_TYPES.values():
                camera_metadata_dict[camera_type] = PinholeCameraMetadata(
                    camera_type=camera_type,
                    width=calibration.width,
                    height=calibration.height,
                    intrinsics=intrinsics,
                    distortion=distortion,
                )

    return camera_metadata_dict


def _get_wopd_lidar_metadata(
    initial_frame: dataset_pb2.Frame,
    keep_polar_features: bool,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LiDARType, LiDARMetadata]:

    laser_metadatas: Dict[LiDARType, LiDARMetadata] = {}

    # NOTE: Using
    lidar_index = WOPDLidarIndex if keep_polar_features else DefaultLidarIndex
    if dataset_converter_config.lidar_store_option is not None:
        for laser_calibration in initial_frame.context.laser_calibrations:

            lidar_type = WOPD_LIDAR_TYPES[laser_calibration.name]

            extrinsic: Optional[StateSE3] = None
            if laser_calibration.extrinsic:
                extrinsic_transform = np.array(laser_calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)
                extrinsic = StateSE3.from_transformation_matrix(extrinsic_transform)

            laser_metadatas[lidar_type] = LiDARMetadata(
                lidar_type=lidar_type,
                lidar_index=lidar_index,
                extrinsic=extrinsic,
            )

    return laser_metadatas


def _get_ego_pose_se3(frame: dataset_pb2.Frame, map_pose_offset: Vector3D) -> StateSE3:
    ego_pose_matrix = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    ego_pose_se3 = StateSE3.from_transformation_matrix(ego_pose_matrix)
    ego_pose_se3.array[StateSE3Index.XYZ] += map_pose_offset.array[Vector3DIndex.XYZ]
    return ego_pose_se3


def _extract_wopd_ego_state(frame: dataset_pb2.Frame, map_pose_offset: Vector3D) -> List[float]:
    rear_axle_pose = _get_ego_pose_se3(frame, map_pose_offset)

    vehicle_parameters = get_wopd_chrysler_pacifica_parameters()
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
    )


def _extract_wopd_box_detections(
    frame: dataset_pb2.Frame, map_pose_offset: Vector3D, zero_roll_pitch: bool = True
) -> BoxDetectionWrapper:

    ego_pose_se3 = _get_ego_pose_se3(frame, map_pose_offset)

    num_detections = len(frame.laser_labels)
    detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    detections_types: List[int] = []
    detections_token: List[str] = []

    for detection_idx, detection in enumerate(frame.laser_labels):

        detection_quaternion = EulerAngles(
            roll=DEFAULT_ROLL,
            pitch=DEFAULT_PITCH,
            yaw=detection.box.heading,
        ).quaternion

        # 2. Fill SE3 Bounding Box
        detections_state[detection_idx, BoundingBoxSE3Index.X] = detection.box.center_x
        detections_state[detection_idx, BoundingBoxSE3Index.Y] = detection.box.center_y
        detections_state[detection_idx, BoundingBoxSE3Index.Z] = detection.box.center_z
        detections_state[detection_idx, BoundingBoxSE3Index.QUATERNION] = detection_quaternion
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
        detections_types.append(WOPD_DETECTION_NAME_DICT[detection.type])
        detections_token.append(str(detection.id))

    detections_state[:, BoundingBoxSE3Index.STATE_SE3] = convert_relative_to_absolute_se3_array(
        origin=ego_pose_se3, se3_array=detections_state[:, BoundingBoxSE3Index.STATE_SE3]
    )
    if zero_roll_pitch:
        euler_array = get_euler_array_from_quaternion_array(detections_state[:, BoundingBoxSE3Index.QUATERNION])
        euler_array[..., EulerAnglesIndex.ROLL] = DEFAULT_ROLL
        euler_array[..., EulerAnglesIndex.PITCH] = DEFAULT_PITCH
        detections_state[..., BoundingBoxSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(euler_array)

    box_detections: List[BoxDetectionSE3] = []
    for detection_idx in range(num_detections):
        box_detections.append(
            BoxDetectionSE3(
                metadata=BoxDetectionMetadata(
                    box_detection_type=detections_types[detection_idx],
                    timepoint=None,
                    track_token=detections_token[detection_idx],
                    confidence=None,
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )

    return BoxDetectionWrapper(box_detections=box_detections)


def _extract_wopd_cameras(
    frame: dataset_pb2.Frame, dataset_converter_config: DatasetConverterConfig
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]]:

    camera_dict: Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]] = {}

    if dataset_converter_config.include_cameras:

        # TODO: Implement option to store images as paths
        assert (
            dataset_converter_config.camera_store_option == "binary"
        ), "Camera store option must be 'binary' for WOPD."

        # NOTE: The extrinsic matrix in frame.context.camera_calibration is fixed to model the ego to camera transformation.
        # The poses in frame.images[idx] are the motion compensated ego poses when the camera triggers.

        camera_extrinsic: Dict[str, StateSE3] = {}
        for calibration in frame.context.camera_calibrations:
            camera_type = WOPD_CAMERA_TYPES[calibration.name]
            camera_transform = np.array(calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)
            camera_pose = StateSE3.from_transformation_matrix(camera_transform)
            # NOTE: WOPD uses a different camera convention than py123d
            # https://arxiv.org/pdf/1912.04838 (Figure 1.)
            camera_pose = convert_camera_convention(
                camera_pose,
                from_convention=CameraConvention.pXpZmY,
                to_convention=CameraConvention.pZmYpX,
            )
            camera_extrinsic[camera_type] = camera_pose

        for image_proto in frame.images:
            camera_type = WOPD_CAMERA_TYPES[image_proto.name]
            camera_bytes: bytes = image_proto.image
            camera_dict[camera_type] = camera_bytes, camera_extrinsic[camera_type]

    return camera_dict


def _extract_wopd_lidars(
    frame: dataset_pb2.Frame,
    keep_polar_features: bool,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LiDARType, npt.NDArray[np.float32]]:

    lidar_data: Dict[LiDARType, npt.NDArray[np.float32]] = {}

    if dataset_converter_config.include_lidars:

        # TODO: Implement option to store point clouds as paths
        assert dataset_converter_config.lidar_store_option == "binary", "Lidar store option must be 'binary' for WOPD."
        (range_images, camera_projections, _, range_image_top_pose) = parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame=frame,
            range_images=range_images,
            camera_projections=camera_projections,
            range_image_top_pose=range_image_top_pose,
            keep_polar_features=keep_polar_features,
        )

        for lidar_idx, frame_lidar in enumerate(frame.lasers):
            lidar_type = WOPD_LIDAR_TYPES[frame_lidar.name]
            lidar_data[lidar_type] = np.array(points[lidar_idx], dtype=np.float32).flatten()

    return lidar_data
