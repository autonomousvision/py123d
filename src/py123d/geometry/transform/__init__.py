"""Coordinate-frame transformations for SE(2) and SE(3) poses and points.

This package exposes functions for converting poses and points between absolute,
relative, and arbitrary reference frames. Two sub-modules are provided:

- :mod:`~py123d.geometry.transform.transform_se2` -- 2D rigid-body transforms
  using ``(x, y, yaw)`` representation.
- :mod:`~py123d.geometry.transform.transform_se3` -- 3D rigid-body transforms
  using ``(x, y, z, qw, qx, qy, qz)`` quaternion representation.

Each function comes in two flavours:

- **Array functions** (suffix ``_array``) operate on raw NumPy arrays and support
  batch dimensions.
- **Typed functions** (no suffix) accept and return typed geometry objects such as
  :class:`~py123d.geometry.PoseSE2`, :class:`~py123d.geometry.PoseSE3`,
  :class:`~py123d.geometry.Point2D`, and :class:`~py123d.geometry.Point3D`.
"""

from py123d.geometry.transform.transform_se2 import (
    # Canonical names: array functions
    abs_to_rel_se2_array,
    rel_to_abs_se2_array,
    reframe_se2_array,
    abs_to_rel_points_2d_array,
    rel_to_abs_points_2d_array,
    reframe_points_2d_array,
    # Canonical names: typed single-item functions
    abs_to_rel_se2,
    rel_to_abs_se2,
    reframe_se2,
    abs_to_rel_point_2d,
    rel_to_abs_point_2d,
    reframe_point_2d,
    # Translation functions
    translate_se2_along_body_frame,
    translate_se2_along_x,
    translate_se2_along_y,
    translate_se2_array_along_body_frame,
    translate_2d_along_body_frame,
    # Deprecated aliases
    convert_absolute_to_relative_se2_array,
    convert_relative_to_absolute_se2_array,
    convert_se2_array_between_origins,
    convert_absolute_to_relative_points_2d_array,
    convert_relative_to_absolute_points_2d_array,
    convert_points_2d_array_between_origins,
)
from py123d.geometry.transform.transform_se3 import (
    # Canonical names: array functions
    abs_to_rel_se3_array,
    rel_to_abs_se3_array,
    reframe_se3_array,
    abs_to_rel_points_3d_array,
    rel_to_abs_points_3d_array,
    reframe_points_3d_array,
    # Canonical names: typed single-item functions
    abs_to_rel_se3,
    rel_to_abs_se3,
    reframe_se3,
    abs_to_rel_point_3d,
    rel_to_abs_point_3d,
    reframe_point_3d,
    # Translation functions
    translate_se3_along_body_frame,
    translate_se3_along_x,
    translate_se3_along_y,
    translate_se3_along_z,
    translate_3d_along_body_frame,
    # Deprecated aliases
    convert_absolute_to_relative_se3_array,
    convert_relative_to_absolute_se3_array,
    convert_se3_array_between_origins,
    convert_absolute_to_relative_points_3d_array,
    convert_relative_to_absolute_points_3d_array,
    convert_points_3d_array_between_origins,
)
