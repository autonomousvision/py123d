from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

_T = TypeVar("_T")

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3
from py123d.geometry.transform import abs_to_rel_points_3d_array


class CameraChannelType(SerialIntEnum):
    """Enumeration of camera channel types."""

    RGB = 0
    GRAYSCALE = 1


class CameraModel(SerialIntEnum):
    """Enumeration of camera projection models."""

    PINHOLE = 0
    """Standard pinhole camera model."""

    FISHEYE_MEI = 1
    """Fisheye camera using the MEI (mirror) model."""

    FTHETA = 2
    """F-theta polynomial camera model."""


class CameraID(SerialIntEnum):
    """Enumeration of camera IDs. These are unique within a sensor rig and can be used as modality IDs for camera metadata."""

    # Pinhole cameras
    # ------------------------------------------------------------------------------------------------------------------

    PCAM_F0 = 0
    """Front pinhole camera."""

    PCAM_B0 = 1
    """Back pinhole camera."""

    PCAM_L0 = 2
    """Left pinhole camera, first from front to back."""

    PCAM_L1 = 3
    """Left pinhole camera, second from front to back."""

    PCAM_L2 = 4
    """Left pinhole camera, third from front to back."""

    PCAM_R0 = 5
    """Right pinhole camera, first from front to back."""

    PCAM_R1 = 6
    """Right pinhole camera, second from front to back."""

    PCAM_R2 = 7
    """Right pinhole camera, third from front to back."""

    PCAM_STEREO_L = 8
    """Left pinhole stereo camera."""

    PCAM_STEREO_R = 9
    """Right pinhole stereo camera."""

    # Fisheye MEI cameras
    # ------------------------------------------------------------------------------------------------------------------

    FMCAM_L = 10
    """Left-facing fisheye MEI camera."""

    FMCAM_R = 11
    """Right-facing fisheye MEI camera."""

    # F-theta cameras
    # ------------------------------------------------------------------------------------------------------------------

    FTCAM_F0 = 12
    """Front F-theta camera."""

    FTCAM_TELE_F0 = 13
    """Front telephoto F-theta camera."""

    FTCAM_TELE_B0 = 18
    """Back telephoto F-theta camera."""

    FTCAM_L0 = 14
    """Left F-theta camera, first from front to back."""

    FTCAM_L1 = 15
    """Left F-theta camera, second from front to back."""

    FTCAM_R0 = 16
    """Right F-theta camera, first from front to back."""

    FTCAM_R1 = 17
    """Right F-theta camera, second from front to back."""


ALL_PINHOLE_CAMERA_IDS = [
    CameraID.PCAM_F0,
    CameraID.PCAM_B0,
    CameraID.PCAM_L0,
    CameraID.PCAM_L1,
    CameraID.PCAM_L2,
    CameraID.PCAM_R0,
    CameraID.PCAM_R1,
    CameraID.PCAM_R2,
    CameraID.PCAM_STEREO_L,
    CameraID.PCAM_STEREO_R,
]

ALL_FISHEYE_MEI_CAMERA_IDS = [
    CameraID.FMCAM_L,
    CameraID.FMCAM_R,
]

ALL_FTHETA_CAMERA_IDS = [
    CameraID.FTCAM_F0,
    CameraID.FTCAM_TELE_F0,
    CameraID.FTCAM_TELE_B0,
    CameraID.FTCAM_L0,
    CameraID.FTCAM_L1,
    CameraID.FTCAM_R0,
    CameraID.FTCAM_R1,
]

# ---------------------------------------------------------------------------
# Camera metadata registry (populated by subclasses via register_camera_metadata)
# ---------------------------------------------------------------------------

_CAMERA_METADATA_REGISTRY: Dict[CameraModel, Type[BaseCameraMetadata]] = {}


def register_camera_metadata(camera_model: CameraModel):
    """Class decorator that registers a BaseCameraMetadata subclass for a given CameraModel."""

    def decorator(cls: _T) -> _T:
        _CAMERA_METADATA_REGISTRY[camera_model] = cls  # type: ignore[assignment]
        return cls

    return decorator


def camera_metadata_from_dict(data_dict: Dict[str, Any]) -> BaseCameraMetadata:
    """Factory function: deserialize a camera metadata dict into the correct subclass.

    Reads the ``"camera_model"`` discriminator field and dispatches to the registered subclass.

    :param data_dict: A dictionary containing the camera metadata with a ``"camera_model"`` field.
    :return: A :class:`BaseCameraMetadata` subclass instance.
    """
    camera_model = CameraModel.from_arbitrary(data_dict["camera_model"])
    if camera_model not in _CAMERA_METADATA_REGISTRY:
        raise ValueError(f"No camera metadata class registered for camera model '{camera_model}'")
    metadata = _CAMERA_METADATA_REGISTRY[camera_model].from_dict(data_dict)
    assert isinstance(metadata, BaseCameraMetadata)
    return metadata


class BaseCameraMetadata(BaseModalityMetadata, abc.ABC):
    """Base class for camera metadata. Provides the shared interface for all camera models."""

    __slots__ = ()

    @property
    @abc.abstractmethod
    def camera_model(self) -> CameraModel:
        """The projection model of the camera."""

    @property
    @abc.abstractmethod
    def camera_id(self) -> CameraID:
        """The camera ID, unique within a sensor rig."""

    @property
    @abc.abstractmethod
    def camera_name(self) -> str:
        """The camera name, according to the dataset naming convention."""

    @property
    @abc.abstractmethod
    def camera_to_imu_se3(self) -> PoseSE3:
        """The static extrinsic pose of the camera relative to the IMU frame."""

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """The width of the camera image in pixels."""

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """The height of the camera image in pixels."""

    @property
    def channel_type(self) -> CameraChannelType:
        """The channel type of the camera image. Defaults to RGB."""
        return CameraChannelType.RGB

    @property
    def modality_type(self) -> ModalityType:
        """Returns the type of the modality that this metadata describes."""
        return ModalityType.CAMERA

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """Returns the camera ID as the modality ID."""
        return self.camera_id

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio (width / height) of the camera."""
        return self.width / self.height

    def _compute_in_fov_mask(
        self,
        pixel_coords: npt.NDArray[np.float64],
        depth: npt.NDArray[np.float64],
        eps: float = 1e-6,
    ) -> npt.NDArray[np.bool_]:
        """Compute a boolean mask for points in front of the camera and within image bounds.

        :param pixel_coords: (N, 2) array of (u, v) pixel coordinates.
        :param depth: (N,) array of depths (positive = in front of camera).
        :param eps: minimum depth threshold, defaults to 1e-6.
        :return: (N,) boolean mask.
        """
        mask = depth > eps
        mask = mask & (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < self.width)
        mask = mask & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < self.height)
        return mask

    @abc.abstractmethod
    def project_to_image(
        self,
        points_cam: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Project 3D points in camera frame to image pixel coordinates.

        :param points_cam: (N, 3) array of 3D points in the camera coordinate frame.
        :return: A tuple of (pixel_coords (N,2), in_fov_mask (N,), depth (N,)).
        """


class Camera(BaseModality):
    """A camera observation: image, extrinsic pose, timestamp, and model-specific metadata."""

    __slots__ = ("_metadata", "_image", "_camera_to_global_se3", "_timestamp")

    def __init__(
        self,
        metadata: BaseCameraMetadata,
        image: npt.NDArray[np.uint8],
        camera_to_global_se3: PoseSE3,
        timestamp: Timestamp,
    ) -> None:
        """Initialize a Camera instance.

        :param metadata: The camera metadata (determines the camera model).
        :param image: The image captured by the camera.
        :param camera_to_global_se3: The extrinsic pose of the camera in global coordinates.
        :param timestamp: The timestamp of the image capture.
        """
        self._metadata = metadata
        self._image = image
        self._camera_to_global_se3 = camera_to_global_se3
        self._timestamp = timestamp

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.time.Timestamp` of the image capture."""
        return self._timestamp

    @property
    def metadata(self) -> BaseCameraMetadata:
        """The :class:`BaseCameraMetadata` associated with the camera."""
        return self._metadata

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """The image captured by the camera, as a numpy array."""
        return self._image

    @property
    def camera_to_global_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the camera in global coordinates."""
        return self._camera_to_global_se3

    def project_points_global(
        self,
        points_global: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Project 3D points in global frame to image pixel coordinates.

        Convenience method that transforms points from global to camera frame,
        then delegates to :meth:`BaseCameraMetadata.project_to_image`.

        :param points_global: (N, 3) array of 3D points in global coordinates.
        :return: A tuple of:
            - pixel_coords: (N, 2) array of (u, v) pixel coordinates.
            - in_fov_mask: (N,) boolean mask.
            - depth: (N,) array of signed depths.
        """
        points_cam = abs_to_rel_points_3d_array(self._camera_to_global_se3, points_global)
        result = self._metadata.project_to_image(points_cam)
        return result
