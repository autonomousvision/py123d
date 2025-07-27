from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from d123.common.utils.enums import SerialIntEnum


class CameraType(SerialIntEnum):
    """
    Enum for cameras in d123.
    """

    CAM_F0 = 0
    CAM_B0 = 1
    CAM_L0 = 2
    CAM_L1 = 3
    CAM_L2 = 4
    CAM_R0 = 5
    CAM_R1 = 6
    CAM_R2 = 7


@dataclass
class CameraMetadata:

    camera_type: CameraType
    width: int
    height: int
    intrinsic: npt.NDArray[np.float64]  # 3x3 matrix
    distortion: npt.NDArray[np.float64]  # 5x1 vector
    translation: npt.NDArray[np.float64]  # 3x1 vector
    rotation: npt.NDArray[np.float64]  # 3x3 matrix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_type": int(self.camera_type),
            "width": self.width,
            "height": self.height,
            "intrinsic": self.intrinsic.tolist(),
            "distortion": self.distortion.tolist(),
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
        }

    @classmethod
    def from_dict(cls, json_dict: Dict[str, Any]) -> CameraMetadata:
        return cls(
            camera_type=CameraType(json_dict["camera_type"]),
            width=json_dict["width"],
            height=json_dict["height"],
            intrinsic=np.array(json_dict["intrinsic"]),
            distortion=np.array(json_dict["distortion"]),
            translation=np.array(json_dict["translation"]),
            rotation=np.array(json_dict["rotation"]),
        )


def camera_metadata_dict_to_json(camera_metadata: Dict[CameraType, CameraMetadata]) -> Dict[str, Dict[str, Any]]:
    """
    Converts a dictionary of CameraMetadata to a JSON-serializable format.
    :param camera_metadata: Dictionary of CameraMetadata.
    :return: JSON-serializable dictionary.
    """
    camera_metadata_dict = {str(camera_type): metadata.to_dict() for camera_type, metadata in camera_metadata.items()}
    return json.dumps(camera_metadata_dict)


def camera_metadata_dict_from_json(json_dict: Dict[str, Dict[str, Any]]) -> Dict[CameraType, CameraMetadata]:
    """
    Converts a JSON-serializable dictionary back to a dictionary of CameraMetadata.
    :param json_dict: JSON-serializable dictionary.
    :return: Dictionary of CameraMetadata.
    """
    camera_metadata_dict = json.loads(json_dict)
    return {
        CameraType.deserialize(camera_type): CameraMetadata.from_dict(metadata)
        for camera_type, metadata in camera_metadata_dict.items()
    }


@dataclass
class Camera:

    metadata: CameraMetadata
    image: npt.NDArray[np.uint8]

    def get_view_matrix(self) -> np.ndarray:
        # Compute the view matrix based on the camera's position and orientation
        pass
