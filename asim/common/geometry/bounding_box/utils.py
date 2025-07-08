import numpy as np
import numpy.typing as npt
import shapely

from asim.common.geometry.base import Point2DIndex
from asim.common.geometry.bounding_box.bounding_box_index import BoundingBoxSE2Index, Corners2DIndex


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


def corners_array_to_polygon_array(corners_array: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
    polygons = shapely.creation.polygons(corners_array)
    return polygons


def bbse2_array_to_polygon_array(bbse2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return corners_array_to_polygon_array(bbse2_array_to_corners_array(bbse2))


def translate_along_yaw_array(
    points_2d: npt.NDArray[np.float64],
    headings: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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
