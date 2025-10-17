from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
from zmq import IntEnum

from py123d.common.utils.enums import SerialIntEnum
from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry.se import StateSE3


class PinholeCameraType(SerialIntEnum):
    """
    Enum for cameras in py123d.
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
class PinholeCamera:

    metadata: PinholeCameraMetadata
    image: npt.NDArray[np.uint8]
    extrinsic: StateSE3


class PinholeIntrinsicsIndex(IntEnum):

    FX = 0
    FY = 1
    CX = 2
    CY = 3
    SKEW = 4  # NOTE: not used, but added for completeness


class PinholeIntrinsics(ArrayMixin):

    _array: npt.NDArray[np.float64]

    def __init__(self, fx: float, fy: float, cx: float, cy: float, skew: float = 0.0) -> None:
        array = np.zeros(len(PinholeIntrinsicsIndex), dtype=np.float64)
        array[PinholeIntrinsicsIndex.FX] = fx
        array[PinholeIntrinsicsIndex.FY] = fy
        array[PinholeIntrinsicsIndex.CX] = cx
        array[PinholeIntrinsicsIndex.CY] = cy
        array[PinholeIntrinsicsIndex.SKEW] = skew
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PinholeIntrinsics:
        assert array.ndim == 1
        assert array.shape[-1] == len(PinholeIntrinsicsIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @classmethod
    def from_camera_matrix(cls, intrinsic: npt.NDArray[np.float64]) -> PinholeIntrinsics:
        """
        Create a PinholeIntrinsics from a 3x3 intrinsic matrix.
        :param intrinsic: A 3x3 numpy array representing the intrinsic matrix.
        :return: A PinholeIntrinsics instance.
        """
        assert intrinsic.shape == (3, 3)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        skew = intrinsic[0, 1]  # Not used in most cases.
        array = np.array([fx, fy, cx, cy, skew], dtype=np.float64)
        return cls.from_array(array, copy=False)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return self._array

    @property
    def fx(self) -> float:
        return self._array[PinholeIntrinsicsIndex.FX]

    @property
    def fy(self) -> float:
        return self._array[PinholeIntrinsicsIndex.FY]

    @property
    def cx(self) -> float:
        return self._array[PinholeIntrinsicsIndex.CX]

    @property
    def cy(self) -> float:
        return self._array[PinholeIntrinsicsIndex.CY]

    @property
    def skew(self) -> float:
        return self._array[PinholeIntrinsicsIndex.SKEW]

    @property
    def camera_matrix(self) -> npt.NDArray[np.float64]:
        """
        Returns the intrinsic matrix.
        :return: A 3x3 numpy array representing the intrinsic matrix.
        """
        K = np.array(
            [
                [self.fx, self.skew, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return K


class PinholeDistortionIndex(IntEnum):
    K1 = 0
    K2 = 1
    P1 = 2
    P2 = 3
    K3 = 4


class PinholeDistortion(ArrayMixin):
    _array: npt.NDArray[np.float64]

    def __init__(self, k1: float, k2: float, p1: float, p2: float, k3: float) -> None:
        array = np.zeros(len(PinholeDistortionIndex), dtype=np.float64)
        array[PinholeDistortionIndex.K1] = k1
        array[PinholeDistortionIndex.K2] = k2
        array[PinholeDistortionIndex.P1] = p1
        array[PinholeDistortionIndex.P2] = p2
        array[PinholeDistortionIndex.K3] = k3
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PinholeDistortion:
        assert array.ndim == 1
        assert array.shape[-1] == len(PinholeDistortionIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return self._array

    @property
    def k1(self) -> float:
        return self._array[PinholeDistortionIndex.K1]

    @property
    def k2(self) -> float:
        return self._array[PinholeDistortionIndex.K2]

    @property
    def p1(self) -> float:
        return self._array[PinholeDistortionIndex.P1]

    @property
    def p2(self) -> float:
        return self._array[PinholeDistortionIndex.P2]

    @property
    def k3(self) -> float:
        return self._array[PinholeDistortionIndex.K3]


@dataclass
class PinholeCameraMetadata:

    camera_type: PinholeCameraType
    intrinsics: Optional[PinholeIntrinsics]
    distortion: Optional[PinholeDistortion]
    width: int
    height: int

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> PinholeCameraMetadata:
        data_dict["camera_type"] = PinholeCameraType(data_dict["camera_type"])
        data_dict["intrinsics"] = (
            PinholeIntrinsics.from_list(data_dict["intrinsics"]) if data_dict["intrinsics"] is not None else None
        )
        data_dict["distortion"] = (
            PinholeDistortion.from_list(data_dict["distortion"]) if data_dict["distortion"] is not None else None
        )
        return PinholeCameraMetadata(**data_dict)

    def to_dict(self) -> Dict[str, Any]:
        data_dict = asdict(self)
        data_dict["camera_type"] = int(self.camera_type)
        data_dict["intrinsics"] = self.intrinsics.tolist() if self.intrinsics is not None else None
        data_dict["distortion"] = self.distortion.tolist() if self.distortion is not None else None
        return data_dict

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def fov_x(self) -> float:
        """
        Calculates the horizontal field of view (FOV) in radian.
        """
        fov_x_rad = 2 * np.arctan(self.width / (2 * self.intrinsics.fx))
        return fov_x_rad

    @property
    def fov_y(self) -> float:
        """
        Calculates the vertical field of view (FOV) in radian.
        """
        fov_y_rad = 2 * np.arctan(self.height / (2 * self.intrinsics.fy))
        return fov_y_rad
