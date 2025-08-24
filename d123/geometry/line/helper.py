import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString

from d123.geometry.base import Point2DIndex, StateSE2Index


def get_linestring_yaws(linestring: LineString) -> npt.NDArray[np.float64]:
    """
    Compute the heading of each coordinate to its successor coordinate. The last coordinate will have the same heading
        as the second last coordinate.
    :param linestring: linestring as a shapely LineString.
    :return: a list of headings associated to each starting coordinate.
    """
    coords: npt.NDArray[np.float64] = np.asarray(linestring.coords, dtype=np.float64)[..., Point2DIndex.XY]
    return get_points_2d_yaws(coords)


def get_points_2d_yaws(points_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert points_array.ndim == 2
    assert points_array.shape[-1] == len(Point2DIndex)
    vectors = np.diff(points_array, axis=0)
    yaws = np.arctan2(vectors.T[1], vectors.T[0])
    yaws = np.append(yaws, yaws[-1])  # pad end with duplicate heading
    assert len(yaws) == len(points_array), "Calculated heading must have the same length as input coordinates"
    return yaws


def get_path_progress(points_array: npt.NDArray[np.float64]) -> list[float]:
    if points_array.shape[-1] == len(Point2DIndex):
        x_diff = np.diff(points_array[..., Point2DIndex.X])
        y_diff = np.diff(points_array[..., Point2DIndex.X])
    elif points_array.shape[-1] == len(StateSE2Index):
        x_diff = np.diff(points_array[..., StateSE2Index.X])
        y_diff = np.diff(points_array[..., StateSE2Index.Y])
    else:
        raise ValueError(
            f"Invalid points_array shape: {points_array.shape}. Expected last dimension to be {len(Point2DIndex)} or "
            f"{len(StateSE2Index)}."
        )
    points_diff: npt.NDArray[np.float64] = np.concatenate(([x_diff], [y_diff]), axis=0, dtype=np.float64)
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff, dtype=np.float64)  # type: ignore
