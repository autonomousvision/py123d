import numpy as np
import numpy.typing as npt

from asim.common.geometry.base import Point2D, StateSE2

# TODO: Refactor 2D and 3D transform functions in a more consistent and general way.


def translate(pose: StateSE2, translation: Point2D) -> StateSE2:
    return StateSE2(pose.x + translation.x, pose.y + translation.y, pose.yaw)


def translate_along_yaw(pose: StateSE2, translation: Point2D) -> StateSE2:
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.array(
        [
            (translation.y * np.cos(pose.yaw + half_pi)) + (translation.x * np.cos(pose.yaw)),
            (translation.y * np.sin(pose.yaw + half_pi)) + (translation.x * np.sin(pose.yaw)),
        ]
    )
    return translate(pose, Point2D.from_array(translation))
