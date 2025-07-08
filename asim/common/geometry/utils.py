import numpy as np

# TODO: move this somewhere else
# TODO: Maybe rename wrap angle?
# TODO: Add implementation for torch, jax, or whatever else is needed.


def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
