from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry import PoseSE3


class PinholeCameraType(SerialIntEnum):
    """Enumeration of pinhole camera types."""

    PCAM_F0 = 0
    """Front camera."""

    PCAM_B0 = 1
    """Back camera."""

    PCAM_L0 = 2
    """Left camera, first from front to back."""

    PCAM_L1 = 3
    """Left camera, second from front to back."""

    PCAM_L2 = 4
    """Left camera, third from front to back."""

    PCAM_R0 = 5
    """Right camera, first from front to back."""

    PCAM_R1 = 6
    """Right camera, second from front to back."""

    PCAM_R2 = 7
    """Right camera, third from front to back."""

    PCAM_STEREO_L = 8
    """Left stereo camera."""

    PCAM_STEREO_R = 9
    """Right stereo camera."""


class PinholeCamera:
    """Represents the recording of a pinhole camera including its metadata, image, and extrinsic pose."""

    __slots__ = ("_metadata", "_image", "_extrinsic")

    def __init__(
        self,
        metadata: PinholeCameraMetadata,
        image: npt.NDArray[np.uint8],
        extrinsic: PoseSE3,
    ) -> None:
        """Initialize a PinholeCamera instance.

        :param metadata: The metadata associated with the camera.
        :param image: The image captured by the camera.
        :param extrinsic: The extrinsic pose of the camera.
        """
        self._metadata = metadata
        self._image = image
        self._extrinsic = extrinsic

    @property
    def metadata(self) -> PinholeCameraMetadata:
        """The static :class:`PinholeCameraMetadata` associated with the pinhole camera."""
        return self._metadata

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """The image captured by the pinhole camera, as a numpy array."""
        return self._image

    @property
    def extrinsic(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the pinhole camera, relative to the ego vehicle frame."""
        return self._extrinsic


class PinholeIntrinsicsIndex(IntEnum):
    """Enumeration of pinhole camera intrinsic parameters."""

    FX = 0
    """Focal length in x direction."""

    FY = 1
    """Focal length in y direction."""

    CX = 2
    """Optical center x coordinate."""

    CY = 3
    """Optical center y coordinate."""

    SKEW = 4
    """Skew coefficient. Not used in most cases."""


class PinholeIntrinsics(ArrayMixin):
    """Pinhole camera intrinsics representation."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, fx: float, fy: float, cx: float, cy: float, skew: float = 0.0) -> None:
        """Initialize PinholeIntrinsics.

        :param fx: Focal length in x direction.
        :param fy: Focal length in y direction.
        :param cx: Optical center x coordinate.
        :param cy: Optical center y coordinate.
        :param skew: Skew coefficient. Not used in most cases, defaults to 0.0
        """
        array = np.zeros(len(PinholeIntrinsicsIndex), dtype=np.float64)
        array[PinholeIntrinsicsIndex.FX] = fx
        array[PinholeIntrinsicsIndex.FY] = fy
        array[PinholeIntrinsicsIndex.CX] = cx
        array[PinholeIntrinsicsIndex.CY] = cy
        array[PinholeIntrinsicsIndex.SKEW] = skew
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PinholeIntrinsics:
        """Creates a PinholeIntrinsics from a numpy array, indexed by :class:`PinholeIntrinsicsIndex`.

        :param array: A 1D numpy array containing the intrinsic parameters.
        :param copy: Whether to copy the array, defaults to True
        :return: A :class:`PinholeIntrinsics` instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(PinholeIntrinsicsIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @classmethod
    def from_camera_matrix(cls, intrinsic: npt.NDArray[np.float64]) -> PinholeIntrinsics:
        """Create a PinholeIntrinsics from a 3x3 intrinsic matrix.

        :param intrinsic: A 3x3 numpy array representing the intrinsic matrix.
        :return: A :class:`PinholeIntrinsics` instance.
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
        """A numpy array representation of the pinhole intrinsics, indexed by :class:`PinholeIntrinsicsIndex`."""
        return self._array

    @property
    def fx(self) -> float:
        """Focal length in x direction."""
        return self._array[PinholeIntrinsicsIndex.FX]

    @property
    def fy(self) -> float:
        """Focal length in y direction."""
        return self._array[PinholeIntrinsicsIndex.FY]

    @property
    def cx(self) -> float:
        """Optical center x coordinate."""
        return self._array[PinholeIntrinsicsIndex.CX]

    @property
    def cy(self) -> float:
        """Optical center y coordinate."""
        return self._array[PinholeIntrinsicsIndex.CY]

    @property
    def skew(self) -> float:
        """Skew coefficient. Not used in most cases."""
        return self._array[PinholeIntrinsicsIndex.SKEW]

    @property
    def camera_matrix(self) -> npt.NDArray[np.float64]:
        """The 3x3 camera intrinsic matrix K."""
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
    """Enumeration of pinhole camera distortion parameters."""

    K1 = 0
    """Radial distortion coefficient k1."""

    K2 = 1
    """Radial distortion coefficient k2."""

    P1 = 2
    """Tangential distortion coefficient p1."""

    P2 = 3
    """Tangential distortion coefficient p2."""

    K3 = 4
    """Radial distortion coefficient k3."""


class PinholeDistortion(ArrayMixin):
    """Pinhole camera distortion representation."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, k1: float, k2: float, p1: float, p2: float, k3: float) -> None:
        """Initialize :class:`:PinholeDistortion`.

        :param k1: Radial distortion coefficient k1.
        :param k2: Radial distortion coefficient k2.
        :param p1: Tangential distortion coefficient p1.
        :param p2: Tangential distortion coefficient p2.
        :param k3: Radial distortion coefficient k3.
        """
        array = np.zeros(len(PinholeDistortionIndex), dtype=np.float64)
        array[PinholeDistortionIndex.K1] = k1
        array[PinholeDistortionIndex.K2] = k2
        array[PinholeDistortionIndex.P1] = p1
        array[PinholeDistortionIndex.P2] = p2
        array[PinholeDistortionIndex.K3] = k3
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PinholeDistortion:
        """Creates a PinholeDistortion from a numpy array, indexed by :class:`PinholeDistortionIndex`.

        :param array: A 1D numpy array containing the distortion parameters.
        :param copy: Whether to copy the array, defaults to True
        :return: A :class:`PinholeDistortion` instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(PinholeDistortionIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """A numpy array representation of the pinhole distortion, indexed by :class:`PinholeDistortionIndex`."""
        return self._array

    @property
    def k1(self) -> float:
        """Radial distortion coefficient k1."""
        return self._array[PinholeDistortionIndex.K1]

    @property
    def k2(self) -> float:
        """Radial distortion coefficient k2."""
        return self._array[PinholeDistortionIndex.K2]

    @property
    def p1(self) -> float:
        """Tangential distortion coefficient p1."""
        return self._array[PinholeDistortionIndex.P1]

    @property
    def p2(self) -> float:
        """Tangential distortion coefficient p2."""
        return self._array[PinholeDistortionIndex.P2]

    @property
    def k3(self) -> float:
        """Radial distortion coefficient k3."""
        return self._array[PinholeDistortionIndex.K3]


class PinholeCameraMetadata:
    """Static metadata for a pinhole camera, stored in a log."""

    __slots__ = ("_camera_type", "_intrinsics", "_distortion", "_width", "_height")

    def __init__(
        self,
        camera_type: PinholeCameraType,
        intrinsics: Optional[PinholeIntrinsics],
        distortion: Optional[PinholeDistortion],
        width: int,
        height: int,
    ) -> None:
        """Initialize a :class:`PinholeCameraMetadata` instance.

        :param camera_type: The type of the pinhole camera.
        :param intrinsics: The :class:`PinholeIntrinsics` of the pinhole camera.
        :param distortion: The :class:`PinholeDistortion` of the pinhole camera.
        :param width: The image width in pixels.
        :param height: The image height in pixels.
        """
        self._camera_type = camera_type
        self._intrinsics = intrinsics
        self._distortion = distortion
        self._width = width
        self._height = height

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> PinholeCameraMetadata:
        """Create a :class:`PinholeCameraMetadata` from a dictionary.

        :param data_dict: A dictionary containing the metadata.
        :return: A PinholeCameraMetadata instance.
        """
        data_dict["camera_type"] = PinholeCameraType(data_dict["camera_type"])
        data_dict["intrinsics"] = (
            PinholeIntrinsics.from_list(data_dict["intrinsics"]) if data_dict["intrinsics"] is not None else None
        )
        data_dict["distortion"] = (
            PinholeDistortion.from_list(data_dict["distortion"]) if data_dict["distortion"] is not None else None
        )
        return PinholeCameraMetadata(**data_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the :class:`PinholeCameraMetadata` to a dictionary.

        :return: A dictionary representation of the PinholeCameraMetadata instance, with default Python types.
        """
        data_dict = {}
        data_dict["camera_type"] = int(self.camera_type)
        data_dict["intrinsics"] = self.intrinsics.tolist() if self.intrinsics is not None else None
        data_dict["distortion"] = self.distortion.tolist() if self.distortion is not None else None
        data_dict["width"] = self.width
        data_dict["height"] = self.height
        return data_dict

    @property
    def camera_type(self) -> PinholeCameraType:
        """The :class:`PinholeCameraType` of the pinhole camera."""
        return self._camera_type

    @property
    def intrinsics(self) -> Optional[PinholeIntrinsics]:
        """The :class:`PinholeIntrinsics` of the pinhole camera."""
        return self._intrinsics

    @property
    def distortion(self) -> Optional[PinholeDistortion]:
        """The :class:`PinholeDistortion` of the pinhole camera."""
        return self._distortion

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self._height

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio (width / height) of the pinhole camera."""
        return self.width / self.height

    @property
    def fov_x(self) -> float:
        """The horizontal field of view (FOV) of the pinhole camera in radians."""
        fov_x_rad = 2 * np.arctan(self.width / (2 * self.intrinsics.fx))
        return fov_x_rad

    @property
    def fov_y(self) -> float:
        """The vertical field of view (FOV) of the pinhole camera in radians."""
        fov_y_rad = 2 * np.arctan(self.height / (2 * self.intrinsics.fy))
        return fov_y_rad
