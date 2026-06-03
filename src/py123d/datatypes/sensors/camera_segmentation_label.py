from __future__ import annotations

import abc
import importlib
from typing import Type

from py123d.common.utils.enums import SerialIntEnum

CAMERA_SEGMENTATION_LABEL_REGISTRY = {}


def register_camera_segmentation_label(enum_class):
    """Decorator to register a :class:`CameraSegmentationLabel` enum class by name."""
    CAMERA_SEGMENTATION_LABEL_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


def resolve_camera_segmentation_label_class(qualified_name: str) -> Type["CameraSegmentationLabel"]:
    """Resolve a :class:`CameraSegmentationLabel` subclass from a registry name or fully qualified path."""
    label_class: Type[CameraSegmentationLabel]
    if qualified_name in CAMERA_SEGMENTATION_LABEL_REGISTRY:
        label_class = CAMERA_SEGMENTATION_LABEL_REGISTRY[qualified_name]
    elif "." in qualified_name:
        module_path, class_name = qualified_name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            label_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import camera segmentation label class: {qualified_name}") from e
    else:
        raise ValueError(f"Unknown camera segmentation label class: {qualified_name}")
    return label_class


class CameraSegmentationLabel(SerialIntEnum):
    """Base class for all per-pixel (2D) semantic segmentation label enums.

    Concrete subclasses live per-dataset in :mod:`py123d.parser.camera_segmentation_registry`, store
    the raw dataset-native class ids, and map to :class:`DefaultCameraSegmentationLabel` via
    :meth:`to_default` — mirroring the :class:`~py123d.datatypes.BoxDetectionLabel` pattern.
    """

    @abc.abstractmethod
    def to_default(self) -> "DefaultCameraSegmentationLabel":
        """Convert to the unified default camera segmentation label."""


@register_camera_segmentation_label
class DefaultCameraSegmentationLabel(CameraSegmentationLabel):
    """Unified per-pixel semantic segmentation taxonomy used in 123D (Cityscapes + sky style)."""

    IGNORE = 0
    """Unlabeled / ego-vehicle / ignore."""

    ROAD = 1
    SIDEWALK = 2
    BUILDING = 3
    POLE = 4
    TRAFFIC_LIGHT = 5
    TRAFFIC_SIGN = 6
    VEGETATION = 7
    TERRAIN = 8
    SKY = 9
    PERSON = 10
    RIDER = 11
    VEHICLE = 12
    TWO_WHEELER = 13

    OTHER = 14
    """Miscellaneous classes that do not fit the above."""

    def to_default(self) -> "DefaultCameraSegmentationLabel":
        """Inherited, see superclass."""
        return self
