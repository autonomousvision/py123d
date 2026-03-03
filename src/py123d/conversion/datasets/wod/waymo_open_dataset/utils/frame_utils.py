# Vendored from waymo-open-dataset v1.6.7
# https://github.com/waymo-research/waymo-open-dataset
# Copyright 2019 The Waymo Open Dataset Authors. Apache License 2.0.
# Modifications: import paths rewritten for vendoring.
"""Utils for Frame protos."""

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from ..protos import dataset_pb2
from . import range_image_utils, transform_utils

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
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, "ZLIB"
                )
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(bytearray(range_image_top_pose_str_tensor.numpy()))

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]

            if len(laser.ri_return1.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
                seg_label_str_tensor = tf.io.decode_compressed(laser.ri_return1.segmentation_label_compressed, "ZLIB")
                seg_label = dataset_pb2.MatrixInt32()
                seg_label.ParseFromString(bytearray(seg_label_str_tensor.numpy()))
                seg_labels[laser.name] = [seg_label]
        if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(laser.ri_return2.range_image_compressed, "ZLIB")
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)

            if len(laser.ri_return2.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
                seg_label_str_tensor = tf.io.decode_compressed(laser.ri_return2.segmentation_label_compressed, "ZLIB")
                seg_label = dataset_pb2.MatrixInt32()
                seg_label.ParseFromString(bytearray(seg_label_str_tensor.numpy()))
                seg_labels[laser.name].append(seg_label)
    return range_images, camera_projections, seg_labels, range_image_top_pose


def convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, ri_index=0, keep_polar_features=False):
    """Convert range images from polar coordinates to Cartesian coordinates."""
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1], range_image_top_pose_tensor[..., 2]
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    for c in frame.context.laser_calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]), height=range_image.shape.dims[0]
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        if keep_polar_features:
            range_image_cartesian = tf.concat([range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

        cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images


def convert_range_image_to_point_cloud(
    frame, range_images, camera_projections, range_image_top_pose, ri_index=0, keep_polar_features=False
):
    """Convert range images to point cloud."""
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []

    cartesian_range_images = convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features
    )

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.compat.v1.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

    return points, cp_points
