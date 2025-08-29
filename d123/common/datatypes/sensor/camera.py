from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Union

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
    CAM_STEREO_L = 8
    CAM_STEREO_R = 9


@dataclass
class CameraMetadata:

    camera_type: CameraType
    width: int
    height: int
    intrinsic: npt.NDArray[np.float64]  # 3x3 matrix # TODO: don't store matrix but values.
    distortion: npt.NDArray[np.float64]  # 5x1 vector # TODO: don't store matrix but values.

    def to_dict(self) -> Dict[str, Any]:
        # TODO: remove None types. Only a placeholder for now.
        return {
            "camera_type": int(self.camera_type),
            "width": self.width,
            "height": self.height,
            "intrinsic": self.intrinsic.tolist() if self.intrinsic is not None else None,
            "distortion": self.distortion.tolist() if self.distortion is not None else None,
        }

    @classmethod
    def from_dict(cls, json_dict: Dict[str, Any]) -> CameraMetadata:
        # TODO: remove None types. Only a placeholder for now.
        return cls(
            camera_type=CameraType(json_dict["camera_type"]),
            width=json_dict["width"],
            height=json_dict["height"],
            intrinsic=np.array(json_dict["intrinsic"]) if json_dict["intrinsic"] is not None else None,
            distortion=np.array(json_dict["distortion"]) if json_dict["distortion"] is not None else None,
        )

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def fov_x(self) -> float:
        """
        Calculates the horizontal field of view (FOV) in radian.
        """
        fx = self.intrinsic[0, 0]
        fov_x_rad = 2 * np.arctan(self.width / (2 * fx))
        return fov_x_rad

    @property
    def fov_y(self) -> float:
        """
        Calculates the vertical field of view (FOV) in radian.
        """
        fy = self.intrinsic[1, 1]
        fov_y_rad = 2 * np.arctan(self.height / (2 * fy))
        return fov_y_rad


@dataclass
class FisheyeMEICameraMetadata:
    camera_type: CameraType
    width: int
    height: int
    mirror_parameters: int
    distortion: npt.NDArray[np.float64]  # k1,k2,p1,p2
    projection_parameters: npt.NDArray[np.float64] #gamma1,gamma2,u0,v0

    def to_dict(self) -> Dict[str, Any]:
        # TODO: remove None types. Only a placeholder for now.
        return {
            "camera_type": int(self.camera_type),
            "width": self.width,
            "height": self.height,
            "mirror_parameters": self.mirror_parameters,
            "distortion": self.distortion.tolist() if self.distortion is not None else None,
            "projection_parameters": self.projection_parameters.tolist() if self.projection_parameters is not None else None,
        }
    
    @classmethod
    def from_dict(cls, json_dict: Dict[str, Any]) -> CameraMetadata:
        # TODO: remove None types. Only a placeholder for now.
        return cls(
            camera_type=CameraType(json_dict["camera_type"]),
            width=json_dict["width"],
            height=json_dict["height"],
            mirror_parameters=json_dict["mirror_parameters"],
            distortion=np.array(json_dict["distortion"]) if json_dict["distortion"] is not None else None,
            projection_parameters=np.array(json_dict["projection_parameters"]) if json_dict["projection_parameters"] is not None else None,
        )

    def cam2image(self, points_3d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ''' camera coordinate to image plane '''
        norm = np.linalg.norm(points_3d, axis=1)

        x = points_3d[:,0] / norm
        y = points_3d[:,1] / norm
        z = points_3d[:,2] / norm

        x /= z+self.mirror_parameters
        y /= z+self.mirror_parameters

        k1 = self.distortion[0]
        k2 = self.distortion[1]
        gamma1 = self.projection_parameters[0]
        gamma2 = self.projection_parameters[1]
        u0 = self.projection_parameters[2]
        v0 = self.projection_parameters[3]

        ro2 = x*x + y*y
        x *= 1 + k1*ro2 + k2*ro2*ro2
        y *= 1 + k1*ro2 + k2*ro2*ro2

        x = gamma1*x + u0
        y = gamma2*y + v0

        return x, y, norm * points_3d[:,2] / np.abs(points_3d[:,2])

def camera_metadata_dict_to_json(camera_metadata: Dict[CameraType, CameraMetadata]) -> Dict[str, Dict[str, Any]]:
    """
    Converts a dictionary of CameraMetadata to a JSON-serializable format.
    :param camera_metadata: Dictionary of CameraMetadata.
    :return: JSON-serializable dictionary.
    """
    camera_metadata_dict = {
        camera_type.serialize(): metadata.to_dict() for camera_type, metadata in camera_metadata.items()
    }
    return json.dumps(camera_metadata_dict)


def camera_metadata_dict_from_json(json_dict: Dict[str, Dict[str, Any]]) -> Dict[CameraType, Union[CameraMetadata, FisheyeMEICameraMetadata]]:
    """
    Converts a JSON-serializable dictionary back to a dictionary of CameraMetadata.
    :param json_dict: JSON-serializable dictionary.
    :return: Dictionary of CameraMetadata.
    """
    camera_metadata_dict = json.loads(json_dict)
    out: Dict[CameraType, Union[CameraMetadata, FisheyeMEICameraMetadata]] = {}
    for camera_type, metadata in camera_metadata_dict.items():
        cam_type = CameraType.deserialize(camera_type)
        if isinstance(metadata, dict) and "mirror_parameters" in metadata:
            out[cam_type] = FisheyeMEICameraMetadata.from_dict(metadata)
        else:
            out[cam_type] = CameraMetadata.from_dict(metadata)
    return out

@dataclass
class Camera:

    metadata: CameraMetadata
    image: npt.NDArray[np.uint8]
    extrinsic: npt.NDArray[np.float64]  # 4x4 matrix

    def get_view_matrix(self) -> np.ndarray:
        # Compute the view matrix based on the camera's position and orientation
        pass
