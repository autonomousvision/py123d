from typing import Dict, List, Tuple

from d123.common.utils.dependencies import check_dependencies

check_dependencies(modules=["tensorflow", "waymo_open_dataset"], optional_name="waymo")
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

RangeImages = Dict["dataset_pb2.LaserName.Name", List[dataset_pb2.MatrixFloat]]
CameraProjections = Dict["dataset_pb2.LaserName.Name", List[dataset_pb2.MatrixInt32]]
SegmentationLabels = Dict["dataset_pb2.LaserName.Name", List[dataset_pb2.MatrixInt32]]
ParsedFrame = Tuple[RangeImages, CameraProjections, SegmentationLabels, dataset_pb2.MatrixFloat]


def parse_range_image_and_camera_projection(frame: dataset_pb2.Frame) -> ParsedFrame:
    """Parse range images and camera projections given a frame.

    Args:
      frame: open dataset frame proto

    Returns:
      range_images: A dict of {laser_name,
        [range_image_first_return, range_image_second_return]}.
      camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
      seg_labels: segmentation labels, a dict of {laser_name,
        [seg_label_first_return, seg_label_second_return]}
      range_image_top_pose: range image pixel pose for top lidar.
    """
    range_images = {}
    camera_projections = {}
    seg_labels = {}
    range_image_top_pose: dataset_pb2.MatrixFloat = dataset_pb2.MatrixFloat()
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(laser.ri_return1.range_image_compressed, "ZLIB")
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(range_image_str_tensor.numpy())
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, "ZLIB"
                )
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(range_image_top_pose_str_tensor.numpy())

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(camera_projection_str_tensor.numpy())
            camera_projections[laser.name] = [cp]

            if len(laser.ri_return1.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
                seg_label_str_tensor = tf.io.decode_compressed(laser.ri_return1.segmentation_label_compressed, "ZLIB")
                seg_label = dataset_pb2.MatrixInt32()
                seg_label.ParseFromString(seg_label_str_tensor.numpy())
                seg_labels[laser.name] = [seg_label]
        if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(laser.ri_return2.range_image_compressed, "ZLIB")
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(range_image_str_tensor.numpy())
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(camera_projection_str_tensor.numpy())
            camera_projections[laser.name].append(cp)

            if len(laser.ri_return2.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
                seg_label_str_tensor = tf.io.decode_compressed(laser.ri_return2.segmentation_label_compressed, "ZLIB")
                seg_label = dataset_pb2.MatrixInt32()
                seg_label.ParseFromString(seg_label_str_tensor.numpy())
                seg_labels[laser.name].append(seg_label)
    return range_images, camera_projections, seg_labels, range_image_top_pose
