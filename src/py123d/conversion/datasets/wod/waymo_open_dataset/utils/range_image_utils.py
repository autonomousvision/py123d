# Vendored from waymo-open-dataset v1.6.7
# https://github.com/waymo-research/waymo-open-dataset
# Copyright 2019 The Waymo Open Dataset Authors. Apache License 2.0.
# Modifications: import paths rewritten for vendoring.
"""Utils to manage range images."""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

__all__ = [
    "compute_range_image_polar",
    "compute_range_image_cartesian",
    "extract_point_cloud_from_range_image",
    "compute_inclination",
]


def _combined_static_and_dynamic_shape(tensor):
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(input=tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def compute_range_image_polar(range_image, extrinsic, inclination, dtype=tf.float32, scope=None):
    _, height, width = _combined_static_and_dynamic_shape(range_image)
    range_image_dtype = range_image.dtype
    range_image = tf.cast(range_image, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    inclination = tf.cast(inclination, dtype=dtype)

    with tf.compat.v1.name_scope(scope, "ComputeRangeImagePolar", [range_image, extrinsic, inclination]):
        with tf.compat.v1.name_scope("Azimuth"):
            az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
            ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - 0.5) / tf.cast(width, dtype=dtype)
            azimuth = (ratios * 2.0 - 1.0) * np.pi - tf.expand_dims(az_correction, -1)

        azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
        inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
        range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image], axis=-1)
        return tf.cast(range_image_polar, dtype=range_image_dtype)


def compute_range_image_cartesian(
    range_image_polar, extrinsic, pixel_pose=None, frame_pose=None, dtype=tf.float32, scope=None
):
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope, "ComputeRangeImageCartesian", [range_image_polar, extrinsic, pixel_pose, frame_pose]
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        range_image_points = tf.stack([x, y, z], -1)
        rotation = extrinsic[..., 0:3, 0:3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        range_image_points = tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        if pixel_pose is not None:
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points) + pixel_pose_translation
            )
            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            range_image_points = (
                tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        return range_image_points


def extract_point_cloud_from_range_image(
    range_image, extrinsic, inclination, pixel_pose=None, frame_pose=None, dtype=tf.float32, scope=None
):
    with tf.compat.v1.name_scope(
        scope, "ExtractPointCloudFromRangeImage", [range_image, extrinsic, inclination, pixel_pose, frame_pose]
    ):
        range_image_polar = compute_range_image_polar(range_image, extrinsic, inclination, dtype=dtype)
        range_image_cartesian = compute_range_image_cartesian(
            range_image_polar, extrinsic, pixel_pose=pixel_pose, frame_pose=frame_pose, dtype=dtype
        )
        return range_image_cartesian


def compute_inclination(inclination_range, height, scope=None):
    with tf.compat.v1.name_scope(scope, "ComputeInclination", [inclination_range]):
        diff = inclination_range[..., 1] - inclination_range[..., 0]
        inclination = (0.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) / tf.cast(
            height, dtype=inclination_range.dtype
        ) * tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1]
        return inclination
