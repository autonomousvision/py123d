import numpy as np
import numpy.typing as npt
import shapely

from d123.geometry.geometry_index import (
    BoundingBoxSE2Index,
    BoundingBoxSE3Index,
    Corners2DIndex,
    Point2DIndex,
)


def bbse2_array_to_corners_array(bbse2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Converts an array of BoundingBoxSE2 objects to a coordinates array.
    :param bbse2: Array of BoundingBoxSE2 objects.
    :return: Coordinates array of shape (n, 5, 2) where n is the number of bounding boxes.
    """
    assert bbse2.shape[-1] == len(BoundingBoxSE2Index)

    ndim_one: bool = bbse2.ndim == 1
    if ndim_one:
        bbse2 = bbse2[None, :]

    corners_array = np.zeros((*bbse2.shape[:-1], len(Corners2DIndex), len(Point2DIndex)), dtype=np.float64)

    centers = bbse2[..., BoundingBoxSE2Index.XY]
    yaws = bbse2[..., BoundingBoxSE2Index.YAW]
    half_length = bbse2[..., BoundingBoxSE2Index.LENGTH] / 2.0
    half_width = bbse2[..., BoundingBoxSE2Index.WIDTH] / 2.0

    corners_array[..., Corners2DIndex.FRONT_LEFT, :] = translate_along_yaw_array(centers, yaws, half_length, half_width)
    corners_array[..., Corners2DIndex.FRONT_RIGHT, :] = translate_along_yaw_array(
        centers, yaws, half_length, -half_width
    )
    corners_array[..., Corners2DIndex.BACK_RIGHT, :] = translate_along_yaw_array(
        centers, yaws, -half_length, -half_width
    )
    corners_array[..., Corners2DIndex.BACK_LEFT, :] = translate_along_yaw_array(centers, yaws, -half_length, half_width)

    return corners_array.squeeze(axis=0) if ndim_one else corners_array


def corners_2d_array_to_polygon_array(corners_array: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
    polygons = shapely.creation.polygons(corners_array)
    return polygons


def bbse2_array_to_polygon_array(bbse2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return corners_2d_array_to_polygon_array(bbse2_array_to_corners_array(bbse2))


def translate_along_yaw_array(
    points_2d: npt.NDArray[np.float64],
    headings: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # TODO: move somewhere else
    assert points_2d.shape[-1] == len(Point2DIndex)
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.stack(
        [
            (lat * np.cos(headings + half_pi)) + (lon * np.cos(headings)),
            (lat * np.sin(headings + half_pi)) + (lon * np.sin(headings)),
        ],
        axis=-1,
    )
    return points_2d + translation


def bbse3_array_to_corners_array(bbse3_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts an array of BoundingBoxSE3 objects to a coordinates array.
    TODO: Fix this function

    :param bbse3_array: Array of BoundingBoxSE3 objects, shape (..., 7) [x, y, z, yaw, pitch, roll, length, width, height].
    :return: Coordinates array of shape (..., 8, 3) where 8 is the number of corners.
    """
    assert bbse3_array.shape[-1] == len(BoundingBoxSE3Index)

    ndim_one: bool = bbse3_array.ndim == 1
    if ndim_one:
        bbse3_array = bbse3_array[None, :]

    # Extract parameters
    centers = bbse3_array[..., BoundingBoxSE3Index.XYZ]  # (..., 3)
    yaws = bbse3_array[..., BoundingBoxSE3Index.YAW]  # (...,)
    pitches = bbse3_array[..., BoundingBoxSE3Index.PITCH]  # (...,)
    rolls = bbse3_array[..., BoundingBoxSE3Index.ROLL]  # (...,)

    # Corner factors: (x, y, z) in box frame
    factors = np.array(
        [
            [+0.5, -0.5, -0.5],  # FRONT_LEFT_BOTTOM
            [+0.5, +0.5, -0.5],  # FRONT_RIGHT_BOTTOM
            [-0.5, +0.5, -0.5],  # BACK_RIGHT_BOTTOM
            [-0.5, -0.5, -0.5],  # BACK_LEFT_BOTTOM
            [+0.5, -0.5, +0.5],  # FRONT_LEFT_TOP
            [+0.5, +0.5, +0.5],  # FRONT_RIGHT_TOP
            [-0.5, +0.5, +0.5],  # BACK_RIGHT_TOP
            [-0.5, -0.5, +0.5],  # BACK_LEFT_TOP
        ],
        dtype=np.float64,
    )  # (8, 3)

    # Box extents
    extents = bbse3_array[..., BoundingBoxSE3Index.EXTENT]  # (...,)
    corners_local = factors[None, :, :] * extents  # (..., 8, 3)

    # Rotation matrices (yaw, pitch, roll)
    def rotation_matrix(yaw, pitch, roll):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    corners_world = np.empty((*bbse3_array.shape[:-1], 8, 3), dtype=np.float64)
    for idx in np.ndindex(bbse3_array.shape[:-1]):
        R = rotation_matrix(yaws[idx], pitches[idx], rolls[idx])
        corners_world[idx] = centers[idx] + (corners_local[idx] @ R.T)

    return corners_world.squeeze(axis=0) if ndim_one else corners_world
