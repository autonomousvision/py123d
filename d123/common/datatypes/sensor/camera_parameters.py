from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class CameraParameters:

    intrinsics: npt.NDArray[np.float64]  # 3x3 matrix
    distortion: npt.NDArray[np.float64]  # 5x1 vector
    translation: npt.NDArray[np.float64]  # 3x1 vector
    rotation: npt.NDArray[np.float64]  # 3x3 matrix


def get_nuplan_camera_parameters() -> CameraParameters:
    # return CameraParameters(focal_length=focal_length, image_size=image_size)
    return CameraParameters(
        intrinsics=np.array(
            [[1.545e03, 0.000e00, 9.600e02], [0.000e00, 1.545e03, 5.600e02], [0.000e00, 0.000e00, 1.000e00]]
        ),
        distortion=np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        translation=np.array([1.66433035e00, -1.32379618e-03, 1.57190200e00]),
        rotation=np.array(
            [
                [-0.00395669, -0.03969443, 0.99920403],
                [-0.99971496, -0.02336898, -0.00488707],
                [0.02354437, -0.99893856, -0.03959065],
            ]
        ),
    )
