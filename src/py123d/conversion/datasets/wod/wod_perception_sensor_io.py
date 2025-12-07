from pathlib import Path
from typing import Dict, Optional

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.datasets.wod.utils.wod_constants import WODP_CAMERA_TYPES, WODP_LIDAR_TYPES
from py123d.conversion.datasets.wod.utils.wod_range_image_utils import parse_range_image_and_camera_projection
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType

check_dependencies(modules=["tensorflow", "waymo_open_dataset"], optional_name="waymo")
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


def _get_frame_at_iteration(filepath: Path, iteration: int) -> Optional[dataset_pb2.Frame]:
    """Helper function to load a Waymo Frame at a specific iteration from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(str(filepath), compression_type="")

    frame: Optional[dataset_pb2.Frame] = None
    for i, data in enumerate(dataset):
        if i == iteration:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(data.numpy())
            break
    return frame


def load_jpeg_binary_from_tf_record_file(
    tf_record_path: Path,
    iteration: int,
    pinhole_camera_type: PinholeCameraType,
) -> Optional[bytes]:
    """Loads the JPEG binary of a specific pinhole camera from a Waymo TFRecord file at a given iteration."""
    frame = _get_frame_at_iteration(tf_record_path, iteration)
    assert frame is not None, f"Frame at iteration {iteration} not found in Waymo file: {tf_record_path}"

    jpeg_binary: Optional[bytes] = None
    for image_proto in frame.images:
        camera_type = WODP_CAMERA_TYPES[image_proto.name]
        if camera_type == pinhole_camera_type:
            jpeg_binary = image_proto.image
            break
    return jpeg_binary


def load_wodp_lidar_pcs_from_file(
    tf_record_path: Path, index: int, keep_polar_features: bool = False
) -> Dict[LiDARType, np.ndarray]:
    """Loads Waymo Open Dataset - Perception (WODP) LiDAR point clouds from a TFRecord file at a given iteration."""

    frame = _get_frame_at_iteration(tf_record_path, index)
    assert frame is not None, f"Frame at iteration {index} not found in Waymo file: {tf_record_path}"
    (range_images, camera_projections, _, range_image_top_pose) = parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame=frame,
        range_images=range_images,
        camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose,
        keep_polar_features=keep_polar_features,
    )
    lidar_pcs_dict: Dict[LiDARType, np.ndarray] = {}
    for lidar_idx, frame_lidar in enumerate(frame.lasers):
        lidar_type = WODP_LIDAR_TYPES[frame_lidar.name]
        lidar_pcs_dict[lidar_type] = np.array(points[lidar_idx], dtype=np.float32)

    return lidar_pcs_dict
