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
    """Metadata for a per-pixel segmentation camera stream (semantic or panoptic/instance).

    A segmentation camera is pixel-aligned to a regular RGB :class:`~py123d.datatypes.Camera`: it
    shares that camera's id, projection model, intrinsics, and extrinsics. This metadata therefore
    *composes* the sibling camera's :class:`BaseCameraMetadata` for all geometry and additionally
    records **which semantic-label taxonomy** the integer class ids use — analogous to how
    :class:`~py123d.datatypes.detections.box_detections_metadata.BoxDetectionsSE3Metadata` records
    its box-detection label class.

    Its :attr:`channel_type` is either :attr:`CameraChannelType.SEMANTIC` (a per-pixel class-id map,
    :attr:`ModalityType.CAMERA_SEGMENTATION`, file ``camera_segmentation.<camera_id>.arrow``) or
    :attr:`CameraChannelType.INSTANCE` (a per-pixel panoptic/instance map,
    :attr:`ModalityType.CAMERA_INSTANCE_SEGMENTATION`, file
    ``camera_instance_segmentation.<camera_id>.arrow``). Either way it is written to its own Arrow
    file, never colliding with the RGB ``camera.<camera_id>``. For an instance stream that packs the
    semantic id (e.g. KITTI-360 ``semanticId * 1000 + instanceId``) the :attr:`segmentation_label_class`
    still documents the semantic component.
    """

    __slots__ = ("_camera_metadata", "_segmentation_label_class", "_channel_type")

    def __init__(
        self,
        camera_metadata: BaseCameraMetadata,
        segmentation_label_class: Type[CameraSegmentationLabel],
        channel_type: CameraChannelType = CameraChannelType.SEMANTIC,
    ) -> None:
        """Initialize a :class:`SegmentationCameraMetadata`.

        :param camera_metadata: The sibling RGB camera metadata providing geometry (id, model,
            intrinsics, extrinsics, resolution).
        :param segmentation_label_class: The dataset-specific :class:`CameraSegmentationLabel` enum
            describing the per-pixel class ids stored in the label map.
        :param channel_type: :attr:`CameraChannelType.SEMANTIC` (default) for a class-id map or
            :attr:`CameraChannelType.INSTANCE` for a panoptic/instance map.
        """
        assert channel_type in (CameraChannelType.SEMANTIC, CameraChannelType.INSTANCE), (
            f"SegmentationCameraMetadata supports SEMANTIC or INSTANCE channels, got {channel_type}."
        )
        self._camera_metadata = camera_metadata
        self._segmentation_label_class = segmentation_label_class
        self._channel_type = channel_type

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
        """:attr:`CameraChannelType.SEMANTIC` or :attr:`CameraChannelType.INSTANCE`."""
        return self._channel_type

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
        """Serialize the metadata, embedding the sibling camera, the label-class path, and channel."""
        label_class = self._segmentation_label_class
        return {
            "camera_metadata": self._camera_metadata.to_dict(),
            "camera_segmentation_label_class": f"{label_class.__module__}.{label_class.__qualname__}",
            "channel_type": self._channel_type.serialize(),
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> SegmentationCameraMetadata:
        """Construct a :class:`SegmentationCameraMetadata` from a dictionary."""
        camera_metadata = camera_metadata_from_dict(data_dict["camera_metadata"])
        segmentation_label_class = resolve_camera_segmentation_label_class(data_dict["camera_segmentation_label_class"])
        # Backward-compatible: logs written before the instance channel existed are semantic-only.
        channel_type = CameraChannelType.deserialize(
            data_dict.get("channel_type", CameraChannelType.SEMANTIC.serialize())
        )
        return cls(
            camera_metadata=camera_metadata,
            segmentation_label_class=segmentation_label_class,
            channel_type=channel_type,
        )

    def __repr__(self) -> str:
        return (
            f"SegmentationCameraMetadata(camera_id={self.camera_id}, "
            f"channel_type={self._channel_type.name}, "
            f"segmentation_label_class={self._segmentation_label_class.__name__})"
        )

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """The camera id, so the stream sits at ``camera_segmentation.<camera_id>``."""
        return self.camera_id
