from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
from zmq import IntEnum

from py123d.common.utils.enums import SerialIntEnum
from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry.se import StateSE3


class FisheyeMEICameraType(SerialIntEnum):
    """
    Enum for fisheye cameras in d123.
    """

    FCAM_L = 0
    FCAM_R = 1


@dataclass
class FisheyeMEICamera:

    metadata: FisheyeMEICameraMetadata
    image: npt.NDArray[np.uint8]
    extrinsic: StateSE3


class FisheyeMEIDistortionIndex(IntEnum):

    K1 = 0
    K2 = 1
    P1 = 2
    P2 = 3


class FisheyeMEIDistortion(ArrayMixin):
    _array: npt.NDArray[np.float64]

    def __init__(self, k1: float, k2: float, p1: float, p2: float) -> None:
        array = np.zeros(len(FisheyeMEIDistortionIndex), dtype=np.float64)
        array[FisheyeMEIDistortionIndex.K1] = k1
        array[FisheyeMEIDistortionIndex.K2] = k2
        array[FisheyeMEIDistortionIndex.P1] = p1
        array[FisheyeMEIDistortionIndex.P2] = p2
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> FisheyeMEIDistortion:
        assert array.ndim == 1
        assert array.shape[-1] == len(FisheyeMEIDistortionIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return self._array

    @property
    def k1(self) -> float:
        return self._array[FisheyeMEIDistortionIndex.K1]

    @property
    def k2(self) -> float:
        return self._array[FisheyeMEIDistortionIndex.K2]

    @property
    def p1(self) -> float:
        return self._array[FisheyeMEIDistortionIndex.P1]

    @property
    def p2(self) -> float:
        return self._array[FisheyeMEIDistortionIndex.P2]


class FisheyeMEIProjectionIndex(IntEnum):

    GAMMA1 = 0
    GAMMA2 = 1
    U0 = 2
    V0 = 3


class FisheyeMEIProjection(ArrayMixin):
    _array: npt.NDArray[np.float64]

    def __init__(self, gamma1: float, gamma2: float, u0: float, v0: float) -> None:
        array = np.zeros(len(FisheyeMEIProjectionIndex), dtype=np.float64)
        array[FisheyeMEIProjectionIndex.GAMMA1] = gamma1
        array[FisheyeMEIProjectionIndex.GAMMA2] = gamma2
        array[FisheyeMEIProjectionIndex.U0] = u0
        array[FisheyeMEIProjectionIndex.V0] = v0
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> FisheyeMEIProjection:
        assert array.ndim == 1
        assert array.shape[-1] == len(FisheyeMEIProjectionIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return self._array

    @property
    def gamma1(self) -> float:
        return self._array[FisheyeMEIProjectionIndex.GAMMA1]

    @property
    def gamma2(self) -> float:
        return self._array[FisheyeMEIProjectionIndex.GAMMA2]

    @property
    def u0(self) -> float:
        return self._array[FisheyeMEIProjectionIndex.U0]

    @property
    def v0(self) -> float:
        return self._array[FisheyeMEIProjectionIndex.V0]


@dataclass
class FisheyeMEICameraMetadata:

    camera_type: FisheyeMEICameraType
    mirror_parameter: Optional[float]
    distortion: Optional[FisheyeMEIDistortion]
    projection: Optional[FisheyeMEIProjection]
    width: int
    height: int

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> FisheyeMEICameraMetadata:
        data_dict["camera_type"] = FisheyeMEICameraType(data_dict["camera_type"])
        data_dict["distortion"] = (
            FisheyeMEIDistortion.from_array(np.array(data_dict["distortion"]))
            if data_dict["distortion"] is not None
            else None
        )
        data_dict["projection"] = (
            FisheyeMEIProjection.from_array(np.array(data_dict["projection"]))
            if data_dict["projection"] is not None
            else None
        )
        return FisheyeMEICameraMetadata(**data_dict)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def to_dict(self) -> Dict[str, Any]:
        data_dict = asdict(self)
        data_dict["camera_type"] = int(self.camera_type)
        data_dict["distortion"] = self.distortion.array.tolist() if self.distortion is not None else None
        data_dict["projection"] = self.projection.array.tolist() if self.projection is not None else None
        return data_dict

    def cam2image(self, points_3d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """camera coordinate to image plane"""
        norm = np.linalg.norm(points_3d, axis=1)

        x = points_3d[:, 0] / norm
        y = points_3d[:, 1] / norm
        z = points_3d[:, 2] / norm

        x /= z + self.mirror_parameter
        y /= z + self.mirror_parameter

        if self.distortion is not None:
            k1 = self.distortion.k1
            k2 = self.distortion.k2
        else:
            k1 = k2 = 0.0

        if self.projection is not None:
            gamma1 = self.projection.gamma1
            gamma2 = self.projection.gamma2
            u0 = self.projection.u0
            v0 = self.projection.v0
        else:
            gamma1 = gamma2 = 1.0
            u0 = v0 = 0.0

        ro2 = x * x + y * y
        x *= 1 + k1 * ro2 + k2 * ro2 * ro2
        y *= 1 + k1 * ro2 + k2 * ro2 * ro2

        x = gamma1 * x + u0
        y = gamma2 * y + v0

        return x, y, norm * points_3d[:, 2] / np.abs(points_3d[:, 2])
