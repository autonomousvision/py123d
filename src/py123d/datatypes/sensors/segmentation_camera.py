from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.sensors.base_camera import (
    BaseCameraMetadata,
    CameraChannelType,
    CameraID,
    CameraModel,
    camera_metadata_from_dict,
)
from py123d.datatypes.sensors.camera_segmentation_label import (
    CameraSegmentationLabel,
    resolve_camera_segmentation_label_class,
)
from py123d.geometry.pose import PoseSE3


class SegmentationCameraMetadata(BaseCameraMetadata):
    """Metadata for a per-pixel semantic segmentation camera stream.

    A segmentation camera is pixel-aligned to a regular RGB :class:`~py123d.datatypes.Camera`: it
    shares that camera's id, projection model, intrinsics, and extrinsics. This metadata therefore
    *composes* the sibling camera's :class:`BaseCameraMetadata` for all geometry and additionally
    records **which semantic-label taxonomy** the integer class ids use — analogous to how
    :class:`~py123d.datatypes.detections.box_detections_metadata.BoxDetectionsSE3Metadata` records
    its box-detection label class.

    Its :attr:`channel_type` is always :attr:`CameraChannelType.SEMANTIC`, so :attr:`modality_type`
    is :attr:`ModalityType.CAMERA_SEGMENTATION` and it is written to its own Arrow file
    (``camera_segmentation.<camera_id>.arrow``), never colliding with the RGB ``camera.<camera_id>``.
    """

    __slots__ = ("_camera_metadata", "_segmentation_label_class")

    def __init__(
        self,
        camera_metadata: BaseCameraMetadata,
        segmentation_label_class: Type[CameraSegmentationLabel],
    ) -> None:
        """Initialize a :class:`SegmentationCameraMetadata`.

        :param camera_metadata: The sibling RGB camera metadata providing geometry (id, model,
            intrinsics, extrinsics, resolution).
        :param segmentation_label_class: The dataset-specific :class:`CameraSegmentationLabel` enum
            describing the per-pixel class ids stored in the label map.
        """
        self._camera_metadata = camera_metadata
        self._segmentation_label_class = segmentation_label_class

    @property
    def camera_metadata(self) -> BaseCameraMetadata:
        """The sibling camera metadata that provides this stream's geometry."""
        return self._camera_metadata

    @property
    def segmentation_label_class(self) -> Type[CameraSegmentationLabel]:
        """The dataset-specific :class:`CameraSegmentationLabel` enum for the stored class ids."""
        return self._segmentation_label_class

    @property
    def channel_type(self) -> CameraChannelType:
        """Always :attr:`CameraChannelType.SEMANTIC` for a segmentation camera."""
        return CameraChannelType.SEMANTIC

    # ------------------------------------------------------------------------------------------------------------------
    # Geometry delegated to the sibling camera metadata
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def camera_model(self) -> CameraModel:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_model

    @property
    def camera_id(self) -> CameraID:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_id

    @property
    def camera_name(self) -> str:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_name

    @property
    def camera_to_imu_se3(self) -> PoseSE3:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_to_imu_se3

    @property
    def width(self) -> int:
        """Inherited, see superclass."""
        return self._camera_metadata.width

    @property
    def height(self) -> int:
        """Inherited, see superclass."""
        return self._camera_metadata.height

    def project_to_image(
        self,
        points_cam: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Inherited, see superclass."""
        return self._camera_metadata.project_to_image(points_cam)

    # ------------------------------------------------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata, embedding the sibling camera and the label-class path."""
        label_class = self._segmentation_label_class
        return {
            "camera_metadata": self._camera_metadata.to_dict(),
            "camera_segmentation_label_class": f"{label_class.__module__}.{label_class.__qualname__}",
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> SegmentationCameraMetadata:
        """Construct a :class:`SegmentationCameraMetadata` from a dictionary."""
        camera_metadata = camera_metadata_from_dict(data_dict["camera_metadata"])
        segmentation_label_class = resolve_camera_segmentation_label_class(data_dict["camera_segmentation_label_class"])
        return cls(camera_metadata=camera_metadata, segmentation_label_class=segmentation_label_class)

    def __repr__(self) -> str:
        return (
            f"SegmentationCameraMetadata(camera_id={self.camera_id}, "
            f"segmentation_label_class={self._segmentation_label_class.__name__})"
        )

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """The camera id, so the stream sits at ``camera_segmentation.<camera_id>``."""
        return self.camera_id
