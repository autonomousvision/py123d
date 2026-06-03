from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import CameraID, LidarFeature
from py123d.parser.wod.utils.wod_constants import WOD_PERCEPTION_CAMERA_IDS, WOD_PERCEPTION_LIDAR_IDS
from py123d.parser.wod.waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection

check_dependencies(modules=["tensorflow"], optional_name="waymo")
import tensorflow as tf

from py123d.parser.wod.waymo_open_dataset.protos import dataset_pb2
from py123d.parser.wod.waymo_open_dataset.utils import frame_utils


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
    pinhole_camera_type: CameraID,
) -> Optional[bytes]:
    """Loads the JPEG binary of a specific pinhole camera from a Waymo TFRecord file at a given iteration."""
    frame = _get_frame_at_iteration(tf_record_path, iteration)
    assert frame is not None, f"Frame at iteration {iteration} not found in Waymo file: {tf_record_path}"

    jpeg_binary: Optional[bytes] = None
    for image_proto in frame.images:
        camera_type = WOD_PERCEPTION_CAMERA_IDS[image_proto.name]
        if camera_type == pinhole_camera_type:
            jpeg_binary = image_proto.image
            break
    return jpeg_binary


def _extract_wod_perception_point_segmentation(
    frame: dataset_pb2.Frame,
    range_images: Dict,
    seg_labels: Dict,
    ri_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts per-point semantic and instance labels aligned with ``convert_range_image_to_point_cloud``.

    The point ordering produced by :func:`frame_utils.convert_range_image_to_point_cloud` iterates the
    laser calibrations sorted by name and keeps points where ``range_image[..., 0] > 0``. This mirrors
    that exact masking and ordering so the returned per-point arrays align 1:1 with the point cloud.

    Waymo only annotates the TOP lidar; lasers without segmentation labels (and the WOD
    ``TYPE_UNDEFINED`` class) get a sentinel value of 0. Per the Waymo spec, each segmentation range
    image has two channels, ``[instance_id, semantic_class]``.

    :return: ``(semantic, instance)`` as ``(N,)`` uint8 / uint16 arrays.
    """
    semantic_arrays: list = []
    instance_arrays: list = []
    for calibration in sorted(frame.context.laser_calibrations, key=lambda c: c.name):
        range_image = range_images[calibration.name][ri_index]
        range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0
        num_points = int(tf.reduce_sum(tf.cast(range_image_mask, tf.int32)).numpy())

        if calibration.name in seg_labels:
            seg_label = seg_labels[calibration.name][ri_index]
            seg_label_tensor = tf.reshape(tf.convert_to_tensor(value=seg_label.data), seg_label.shape.dims)
            point_labels = tf.gather_nd(seg_label_tensor, tf.compat.v1.where(range_image_mask)).numpy()
            instance_arrays.append(point_labels[:, 0].astype(np.uint16))
            semantic_arrays.append(point_labels[:, 1].astype(np.uint8))
        else:
            instance_arrays.append(np.zeros(num_points, dtype=np.uint16))
            semantic_arrays.append(np.zeros(num_points, dtype=np.uint8))

    return np.concatenate(semantic_arrays, axis=0), np.concatenate(instance_arrays, axis=0)


def load_wod_perception_camera_semantic_label(camera_segmentation_label) -> Optional[np.ndarray]:
    """Decodes a WOD-Perception 2D camera semantic class-id label map, or ``None`` if unannotated.

    Waymo stores a panoptic label PNG where ``panoptic_label = semantic * panoptic_label_divisor +
    instance``. We decode it (as Waymo does, uint16) and recover the per-pixel semantic class ids; the
    raw ids correspond to :class:`~py123d.parser.camera_segmentation_registry.WODPerceptionCameraSegmentationLabel`.

    :return: A ``(H, W)`` uint8 array of semantic class ids, or ``None`` if no segmentation label exists.
    """
    semantic_label: Optional[np.ndarray] = None
    if len(camera_segmentation_label.panoptic_label) > 0:
        divisor = int(camera_segmentation_label.panoptic_label_divisor)
        panoptic_label = tf.image.decode_png(
            camera_segmentation_label.panoptic_label, channels=1, dtype=tf.uint16
        ).numpy()[..., 0]
        semantic_label = (panoptic_label.astype(np.int64) // divisor).astype(np.uint8)
    return semantic_label


def load_wod_perception_point_cloud_data_from_frame(
    frame: dataset_pb2.Frame,
    keep_polar_features: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads Waymo Open Dataset (WOD) - Perception Lidar point clouds from a Waymo Frame object."""

    (range_images, camera_projections, seg_labels, range_image_top_pose) = parse_range_image_and_camera_projection(
        frame
    )
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame=frame,
        range_images=range_images,
        camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose,
        keep_polar_features=keep_polar_features,
    )
    # NOTE: @DanielDauner
    # keep_polar_features=True: points have shape (N, 6) with features in order [RANGE, INTENSITY, ELONGATION, X, Y, Z]
    # keep_polar_features=False: points have shape (N, 3) with features in order [X, Y, Z]

    # Concat all lidar points.
    all_lidar_data = np.concatenate(points, axis=0)

    # Load features and point cloud
    lidar_ids = np.zeros(all_lidar_data.shape[0], dtype=np.uint8)
    start_idx = 0
    for lidar_idx, frame_lidar in enumerate(frame.lasers):
        lidar_id = WOD_PERCEPTION_LIDAR_IDS[frame_lidar.name]
        num_points = points[lidar_idx].shape[0]
        lidar_ids[start_idx : start_idx + num_points] = int(lidar_id)  # type: ignore
        start_idx += num_points

    # Load point cloud and other features based on whether to keep polar features or not.
    if keep_polar_features:
        point_cloud_3d = all_lidar_data[:, 3:6]  # Extract XYZ from the concatenated Lidar data.
        point_cloud_features = {
            LidarFeature.RANGE.serialize(): all_lidar_data[:, 0].astype(np.float32),
            LidarFeature.INTENSITY.serialize(): (all_lidar_data[:, 1] * 255).astype(np.uint8),
            LidarFeature.ELONGATION.serialize(): all_lidar_data[:, 2].astype(np.float32),
            LidarFeature.IDS.serialize(): lidar_ids,
        }
    else:
        point_cloud_3d = all_lidar_data[:, :3]  # Extract XYZ from the concatenated Lidar data.
        point_cloud_features = {
            LidarFeature.IDS.serialize(): lidar_ids,
        }

    # Per-point 3D semantic segmentation, only on frames Waymo annotated (sparse). The TOP-only seg
    # labels are placed at their points; all other points keep the sentinel class 0 (TYPE_UNDEFINED).
    if seg_labels:
        semantic, instance = _extract_wod_perception_point_segmentation(frame, range_images, seg_labels)
        assert semantic.shape[0] == point_cloud_3d.shape[0], (
            "Per-point segmentation labels are misaligned with the point cloud."
        )
        point_cloud_features[LidarFeature.SEMANTIC.serialize()] = semantic
        point_cloud_features[LidarFeature.INSTANCE.serialize()] = instance

    return point_cloud_3d, point_cloud_features


def load_wod_perception_point_cloud_data_from_path(
    tf_record_path: Path,
    index: int,
    keep_polar_features: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads Waymo Open Dataset (WOD) - Perception Lidar point clouds from a TFRecord file at a given iteration."""

    frame = _get_frame_at_iteration(tf_record_path, index)
    assert frame is not None, f"Frame at iteration {index} not found in Waymo file: {tf_record_path}"
    return load_wod_perception_point_cloud_data_from_frame(frame, keep_polar_features=keep_polar_features)
