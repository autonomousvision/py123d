import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE2Index

SECONDS_PER_ITERATION = 0.1


def _masked_diff(y: npt.NDArray[np.float64], mask: npt.NDArray[np.bool], axis: int = 1) -> npt.NDArray[np.float64]:
    """
    Computes the difference between successive elements of y, applying the mask.
    :param y: The input array.
    :param mask: A boolean mask indicating valid elements.
    :return: An array of differences with the same shape as y.
    """

    diff = np.zeros_like(y, dtype=np.float64)
    diff[:, 1:] = np.diff(y, axis=axis)
    diff[:, 0] = diff[:, 1]
    diff[~mask] = 0.0

    return diff


def _get_linear_speed_from_agents_array(
    agents_array: npt.NDArray[np.float64], mask: npt.NDArray[np.bool]
) -> npt.NDArray[np.float64]:
    """
    Extracts the linear speed from the agents array.
    :param agents_array: The agents array containing bounding box data.
    :return: An array of linear speeds.
    """
    assert agents_array.ndim == 3
    assert agents_array.shape[-1] == len(BoundingBoxSE2Index)

    n_agents, n_iterations = agents_array.shape[:2]
    linear_speed = np.zeros((n_agents, n_iterations), dtype=np.float64)
    linear_speed[:, 1:] = (
        np.linalg.norm(np.diff(agents_array[:, :, BoundingBoxSE2Index.XY], axis=1), axis=-1) / SECONDS_PER_ITERATION
    )
    linear_speed[:, 0] = linear_speed[:, 1]
    linear_speed[~mask] = 0.0

    return linear_speed


def _get_linear_acceleration_from_agents_array(
    agents_array: npt.NDArray[np.float64], mask: npt.NDArray[np.bool]
) -> npt.NDArray[np.float64]:
    """
    Extracts the linear acceleration from the agents array.
    :param agents_array: The agents array containing bounding box data.
    :return: An array of linear accelerations.
    """
    assert agents_array.ndim == 3
    assert agents_array.shape[-1] == len(BoundingBoxSE2Index)

    n_agents, n_iterations = agents_array.shape[:2]
    linear_acceleration = np.zeros((n_agents, n_iterations), dtype=np.float64)

    linear_speed = _get_linear_speed_from_agents_array(agents_array, mask)
    linear_acceleration[:, 1:] = np.diff(linear_speed, axis=1) / SECONDS_PER_ITERATION
    linear_acceleration[:, 0] = linear_acceleration[:, 1]
    linear_acceleration[~mask] = 0.0

    return linear_acceleration


def _get_yaw_rate_from_agents_array(
    agents_array: npt.NDArray[np.float64], mask: npt.NDArray[np.bool]
) -> npt.NDArray[np.float64]:
    """
    Extracts the yaw rate from the agents array.
    :param agents_array: The agents array containing bounding box data.
    :param mask: A boolean mask indicating valid elements.
    :return: An array of yaw rates.
    """
    assert agents_array.ndim == 3
    assert agents_array.shape[-1] == len(BoundingBoxSE2Index)

    n_agents, n_iterations = agents_array.shape[:2]
    headings = agents_array[:, :, BoundingBoxSE2Index.YAW]
    heading_rate = _phase_unwrap(_masked_diff(headings, mask, axis=1)) / SECONDS_PER_ITERATION
    return heading_rate


def _get_yaw_acceleration_from_agents_array(
    agents_array: npt.NDArray[np.float64], mask: npt.NDArray[np.bool]
) -> npt.NDArray[np.float64]:
    assert agents_array.ndim == 3
    assert agents_array.shape[-1] == len(BoundingBoxSE2Index)

    n_agents, n_iterations = agents_array.shape[:2]
    yaw_rate = _get_yaw_rate_from_agents_array(agents_array, mask)
    yaw_acceleration = np.zeros((n_agents, n_iterations), dtype=np.float64)
    yaw_acceleration[:, 1:] = np.diff(yaw_rate, axis=1) / SECONDS_PER_ITERATION
    yaw_acceleration[:, 0] = yaw_acceleration[:, 1]
    yaw_acceleration[~mask] = 0.0
    return yaw_acceleration


def _phase_unwrap(headings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _approximate_derivatives(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative
