"""
Default Camera Coordinate System in 123D:

     -Y (up)            /| H
      |                / | e
      |               /  | i
      |              /   | g
      |             /    | h
      |            |     | t
      O────────────|──●──|──────────── +Z (forward), aka. optical/principal axis
     /             |    / h
    /              |   / t
   /               |  / d
  /                | / i
+X (right)         |/ W

We use COLMAP/OpenCV convention (+Z forward, -Y up, +X right),
    abbreviated as "pZmYpX" for the forward-up-right axes.

Other common conventions include, for forward-up-right axes.
 - (+X forward, +Z up, -Y right), "pXpZmY", e.g. Waymo Open Dataset

NOTE: This file should be extended if other conventions are needed in the future.
"""

from enum import Enum
from typing import Union

import numpy as np

from py123d.geometry import PoseSE3


class CameraConvention(Enum):
    """Camera coordinate system conventions
    p/m: positive/negative
    X/Y/Z: axes directions
    order: forward, up, right

    Example: pZmYpX means +Z forward, -Y up, +X right

    TODO: Figure out a more intuitive naming scheme.
    """

    pZmYpX = "pZmYpX"  # Default in 123D (OpenCV/COLMAP)
    pXpZmY = "pXpZmY"  # e.g. Waymo Open Dataset


def convert_camera_convention(
    from_pose: PoseSE3,
    from_convention: Union[CameraConvention, str],
    to_convention: Union[CameraConvention, str],
) -> PoseSE3:
    """Convert camera pose between different conventions.
    123D default is pZmYpX (+Z forward, -Y up, +X right).

    :param from_pose: PoseSE3 representing the camera pose to convert
    :param from_convention: CameraConvention representing the current convention of the pose
    :param to_convention: CameraConvention representing the target convention to convert to
    :return: PoseSE3 representing the converted camera pose
    """
    # TODO: Write tests for this function
    # TODO: Create function over batch/array of poses

    if isinstance(from_convention, str):
        from_convention = CameraConvention(from_convention)
    if isinstance(to_convention, str):
        to_convention = CameraConvention(to_convention)

    if from_convention == to_convention:
        return from_pose

    flip_matrices = {
        (CameraConvention.pXpZmY, CameraConvention.pZmYpX): np.array(
            [
                [0.0, -1.0, 0.0],  # new X = old -Y (right)
                [0.0, 0.0, -1.0],  # new Y = old -Z (down)
                [1.0, 0.0, 0.0],  # new Z = old X (forward)
            ],
            dtype=np.float64,
        ).T,
        (CameraConvention.pZmYpX, CameraConvention.pXpZmY): np.array(
            [
                [0.0, 0.0, 1.0],  # new X = old Z (right)
                [-1.0, 0.0, 0.0],  # new Y = old -X (down)
                [0.0, -1.0, 0.0],  # new Z = old -Y (forward)
            ],
            dtype=np.float64,
        ).T,
    }

    pose_transformation = from_pose.transformation_matrix.copy()
    F = flip_matrices[(from_convention, to_convention)]
    pose_transformation[:3, :3] = pose_transformation[:3, :3] @ F
    return PoseSE3.from_transformation_matrix(pose_transformation)
