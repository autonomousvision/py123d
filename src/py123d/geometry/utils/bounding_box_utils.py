from typing import Tuple

import numpy as np
import numpy.typing as npt
import shapely

from py123d.geometry.geometry_index import (
    BoundingBoxSE2Index,
    BoundingBoxSE3Index,
    Corners2DIndex,
    Corners3DIndex,
    Point2DIndex,
    Point3DIndex,
    Vector2DIndex,
    Vector3DIndex,
)
from py123d.geometry.transform.transform_se2 import translate_2d_along_body_frame
from py123d.geometry.transform.transform_se3 import translate_3d_along_body_frame


def get_corners_2d_factors() -> npt.NDArray[np.float64]:
    """Returns the factors to compute the corners of a SE2 bounding box in the body frame.

    The factors are defined such that multiplying them with the length and width
    of the bounding box yields the corner coordinates in the body frame.

    :return: A (4, 2), indexed by :class:`~py123d.geometry.Corners2DIndex` and
        :class:`~py123d.geometry.Point2DIndex`, respectively.
    """
    # NOTE: ISO 8855 convention for rotation
    factors = np.zeros((len(Corners2DIndex), len(Point2DIndex)), dtype=np.float64)
    factors.fill(0.5)
    factors[Corners2DIndex.FRONT_LEFT] *= [+1, +1]
    factors[Corners2DIndex.FRONT_RIGHT] *= [+1, -1]
    factors[Corners2DIndex.BACK_RIGHT] *= [-1, -1]
    factors[Corners2DIndex.BACK_LEFT] *= [-1, +1]
    return factors


def get_corners_3d_factors() -> npt.NDArray[np.float64]:
    """Returns the factors to compute the corners of a SE3 bounding box in the body frame.

    The factors are defined such that multiplying them with the length, width, and height
    of the bounding box yields the corner coordinates in the body frame.

    :return: A (8, 3), indexed by :class:`~py123d.geometry.Corners3DIndex` and
        :class:`~py123d.geometry.Vector3DIndex`, respectively.
    """
    # NOTE: ISO 8855 convention for rotation
    factors = np.zeros((len(Corners3DIndex), len(Vector3DIndex)), dtype=np.float64)
    factors.fill(0.5)
    factors[Corners3DIndex.FRONT_LEFT_BOTTOM] *= [+1, +1, -1]
    factors[Corners3DIndex.FRONT_RIGHT_BOTTOM] *= [+1, -1, -1]
    factors[Corners3DIndex.BACK_RIGHT_BOTTOM] *= [-1, -1, -1]
    factors[Corners3DIndex.BACK_LEFT_BOTTOM] *= [-1, +1, -1]
    factors[Corners3DIndex.FRONT_LEFT_TOP] *= [+1, +1, +1]
    factors[Corners3DIndex.FRONT_RIGHT_TOP] *= [+1, -1, +1]
    factors[Corners3DIndex.BACK_RIGHT_TOP] *= [-1, -1, +1]
    factors[Corners3DIndex.BACK_LEFT_TOP] *= [-1, +1, +1]
    return factors


def bbse2_array_to_corners_array(bbse2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts an array of BoundingBoxSE2 objects to the 2D coordinates array of their corners.

    :param bbse2: Array of SE2 bounding boxes, indexed by :class:`~py123d.geometry.BoundingBoxSE2Index`.
    :return: Coordinates array of shape (..., 4, 2), indexed by
        :class:`~py123d.geometry.Corners2DIndex` and :class:`~py123d.geometry.Point2DIndex`, respectively.
    """
    assert bbse2.shape[-1] == len(BoundingBoxSE2Index)

    ndim_one: bool = bbse2.ndim == 1
    if ndim_one:
        bbse2 = bbse2[None, ...]

    centers = bbse2[..., BoundingBoxSE2Index.XY]
    yaws = bbse2[..., BoundingBoxSE2Index.YAW]
    extents = bbse2[..., BoundingBoxSE2Index.EXTENT]  # (..., 2)

    factors = get_corners_2d_factors()  # (4, 2)
    corner_translation_body = extents[..., None, :] * factors[None, :, :]  # (..., 4, 2)

    corners_array = translate_2d_along_body_frame(  # (..., 4, 2)
        points_2d=centers[..., None, :],  # (..., 1, 2)
        yaws=yaws[..., None],  # (..., 1)
        x_translate=corner_translation_body[..., Vector2DIndex.X],
        y_translate=corner_translation_body[..., Vector2DIndex.Y],
    )  # (..., 4, 2)

    return corners_array.squeeze(axis=0) if ndim_one else corners_array


def corners_2d_array_to_polygon_array(corners_array: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
    """Converts an array of 2D corners to an array of shapely Polygons.
    TODO: Consider removing this function?

    :param corners_array: Array of shape (..., 4, 2) where 4 is the number of corners.
    :return: Array of shapely Polygons.
    """
    return shapely.creation.polygons(corners_array)  # type: ignore


def bbse2_array_to_polygon_array(bbse2: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
    """Converts an array of BoundingBoxSE2 objects to an array of shapely Polygons.

    :param bbse2: Array of SE2 bounding boxes, indexed by :class:`~py123d.geometry.BoundingBoxSE2Index`.
    :return: Array of shapely Polygons.
    """
    return corners_2d_array_to_polygon_array(bbse2_array_to_corners_array(bbse2))


def bbse3_array_to_corners_array(bbse3_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts an array of BoundingBoxSE3 objects to the 3D coordinates array of their corners.

    :param bbse3_array: Array of SE3 bounding boxes, indexed by :class:`~py123d.geometry.BoundingBoxSE3Index`.
    :return: Coordinates array of shape (..., 8, 3), indexed by
        :class:`~py123d.geometry.Corners3DIndex` and :class:`~py123d.geometry.Point3DIndex`, respectively.
    """
    assert bbse3_array.shape[-1] == len(BoundingBoxSE3Index)

    # Flag whether to unsqueeze and squeeze the input dim
    ndim_one: bool = bbse3_array.ndim == 1
    if ndim_one:
        bbse3_array = bbse3_array[None, ...]

    # Extract parameters
    centers = bbse3_array[..., BoundingBoxSE3Index.XYZ]  # (..., 3)
    quaternions = bbse3_array[..., BoundingBoxSE3Index.QUATERNION]  # (..., 4)

    # Box extents
    factors = get_corners_3d_factors()  # (8, 3)
    extents = bbse3_array[..., BoundingBoxSE3Index.EXTENT]  # (..., 3)
    corner_translation_body = extents[..., None, :] * factors[None, :, :]  # (..., 8, 3)
    corners_world = translate_3d_along_body_frame(
        centers[..., None, :],  # (..., 1, 3)
        quaternions[..., None, :],  # (..., 1, 4)
        corner_translation_body,
    )

    return corners_world.squeeze(axis=0) if ndim_one else corners_world


def corners_array_to_3d_mesh(
    corners_array: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Creates a triangular mesh representation of boxes defined by their corner points.

    :param corners_array: An array of shape (..., 8, 3) representing the corners of the boxes.
    :return: A tuple containing the vertices and faces of the mesh.
    """

    num_boxes = corners_array.shape[0]
    vertices = corners_array.reshape(-1, 3)

    # Define the faces for a single box using Corners3DIndex
    faces_single = np.array(
        [
            # Bottom face
            [Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.FRONT_RIGHT_BOTTOM, Corners3DIndex.BACK_RIGHT_BOTTOM],
            [Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.BACK_RIGHT_BOTTOM, Corners3DIndex.BACK_LEFT_BOTTOM],
            # Top face
            [Corners3DIndex.BACK_RIGHT_TOP, Corners3DIndex.FRONT_RIGHT_TOP, Corners3DIndex.FRONT_LEFT_TOP],
            [Corners3DIndex.BACK_LEFT_TOP, Corners3DIndex.BACK_RIGHT_TOP, Corners3DIndex.FRONT_LEFT_TOP],
            # Left face
            [Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.BACK_LEFT_BOTTOM, Corners3DIndex.BACK_LEFT_TOP],
            [Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.BACK_LEFT_TOP, Corners3DIndex.FRONT_LEFT_TOP],
            # Right face
            [Corners3DIndex.BACK_RIGHT_TOP, Corners3DIndex.BACK_RIGHT_BOTTOM, Corners3DIndex.FRONT_RIGHT_BOTTOM],
            [Corners3DIndex.FRONT_RIGHT_TOP, Corners3DIndex.BACK_RIGHT_TOP, Corners3DIndex.FRONT_RIGHT_BOTTOM],
            # Front face
            [Corners3DIndex.FRONT_RIGHT_TOP, Corners3DIndex.FRONT_RIGHT_BOTTOM, Corners3DIndex.FRONT_LEFT_BOTTOM],
            [Corners3DIndex.FRONT_LEFT_TOP, Corners3DIndex.FRONT_RIGHT_TOP, Corners3DIndex.FRONT_LEFT_BOTTOM],
            # Back face
            [Corners3DIndex.BACK_LEFT_TOP, Corners3DIndex.BACK_LEFT_BOTTOM, Corners3DIndex.BACK_RIGHT_BOTTOM],
            [Corners3DIndex.BACK_RIGHT_TOP, Corners3DIndex.BACK_LEFT_TOP, Corners3DIndex.BACK_RIGHT_BOTTOM],
        ],
        dtype=np.int32,
    )

    # Offset the faces for each box
    faces = np.vstack([faces_single + i * len(Corners3DIndex) for i in range(num_boxes)])

    return vertices, faces


def corners_array_to_edge_lines(corners_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Creates line segments representing the edges of boxes defined by their corner points.

    :param corners_array: An array of shape (..., 8, 3) representing the corners of the boxes.
    :return: An array of shape (..., 12, 2, 3) representing the edge lines of the boxes.
    """

    assert corners_array.shape[-1] == len(Point3DIndex)
    assert corners_array.shape[-2] == len(Corners3DIndex)
    assert corners_array.ndim >= 2

    index_pairs = [
        (Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.FRONT_RIGHT_BOTTOM),
        (Corners3DIndex.FRONT_RIGHT_BOTTOM, Corners3DIndex.BACK_RIGHT_BOTTOM),
        (Corners3DIndex.BACK_RIGHT_BOTTOM, Corners3DIndex.BACK_LEFT_BOTTOM),
        (Corners3DIndex.BACK_LEFT_BOTTOM, Corners3DIndex.FRONT_LEFT_BOTTOM),
        (Corners3DIndex.FRONT_LEFT_TOP, Corners3DIndex.FRONT_RIGHT_TOP),
        (Corners3DIndex.FRONT_RIGHT_TOP, Corners3DIndex.BACK_RIGHT_TOP),
        (Corners3DIndex.BACK_RIGHT_TOP, Corners3DIndex.BACK_LEFT_TOP),
        (Corners3DIndex.BACK_LEFT_TOP, Corners3DIndex.FRONT_LEFT_TOP),
        (Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.FRONT_LEFT_TOP),
        (Corners3DIndex.FRONT_RIGHT_BOTTOM, Corners3DIndex.FRONT_RIGHT_TOP),
        (Corners3DIndex.BACK_RIGHT_BOTTOM, Corners3DIndex.BACK_RIGHT_TOP),
        (Corners3DIndex.BACK_LEFT_BOTTOM, Corners3DIndex.BACK_LEFT_TOP),
    ]

    if corners_array.ndim == 2:
        # Single box case: (8, 3)
        edge_lines = np.zeros((len(index_pairs), 2, len(Point3DIndex)), dtype=np.float64)
        for edge_idx, (start_idx, end_idx) in enumerate(index_pairs):
            edge_lines[edge_idx, 0] = corners_array[start_idx]
            edge_lines[edge_idx, 1] = corners_array[end_idx]
    else:
        # Batched case: (..., 8, 3)
        batch_shape = corners_array.shape[:-2]
        edge_lines = np.zeros(batch_shape + (len(index_pairs), 2, len(Point3DIndex)), dtype=np.float64)
        for edge_idx, (start_idx, end_idx) in enumerate(index_pairs):
            edge_lines[..., edge_idx, 0, :] = corners_array[..., start_idx, :]
            edge_lines[..., edge_idx, 1, :] = corners_array[..., end_idx, :]

    return edge_lines
