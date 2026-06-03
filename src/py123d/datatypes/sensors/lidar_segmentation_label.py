from __future__ import annotations

import abc
import importlib
from typing import Type

from py123d.common.utils.enums import SerialIntEnum

LIDAR_SEGMENTATION_LABEL_REGISTRY = {}


def register_lidar_segmentation_label(enum_class):
    """Decorator to register a :class:`LidarSegmentationLabel` enum class by name."""
    LIDAR_SEGMENTATION_LABEL_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


def resolve_lidar_segmentation_label_class(qualified_name: str) -> Type["LidarSegmentationLabel"]:
    """Resolve a :class:`LidarSegmentationLabel` subclass from a registry name or fully qualified path."""
    label_class: Type[LidarSegmentationLabel]
    if qualified_name in LIDAR_SEGMENTATION_LABEL_REGISTRY:
        label_class = LIDAR_SEGMENTATION_LABEL_REGISTRY[qualified_name]
    elif "." in qualified_name:
        module_path, class_name = qualified_name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            label_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import lidar segmentation label class: {qualified_name}") from e
    else:
        raise ValueError(f"Unknown lidar segmentation label class: {qualified_name}")
    return label_class


class LidarSegmentationLabel(SerialIntEnum):
    """Base class for all per-point (3D) semantic segmentation label enums.

    Concrete subclasses live per-dataset in :mod:`py123d.parser.lidar_segmentation_registry`, store
    the raw dataset-native class ids, and map to :class:`DefaultLidarSegmentationLabel` via
    :meth:`to_default` — mirroring the :class:`~py123d.datatypes.BoxDetectionLabel` pattern.
    """

    @abc.abstractmethod
    def to_default(self) -> "DefaultLidarSegmentationLabel":
        """Convert to the unified default lidar segmentation label."""


@register_lidar_segmentation_label
class DefaultLidarSegmentationLabel(LidarSegmentationLabel):
    """Unified per-point semantic segmentation taxonomy used in 123D (nuScenes-lidarseg style)."""

    IGNORE = 0
    """Unlabeled / noise / ignore."""

    VEHICLE = 1
    TWO_WHEELER = 2
    PERSON = 3
    RIDER = 4

    DRIVABLE_SURFACE = 5
    SIDEWALK = 6
    TERRAIN = 7
    VEGETATION = 8
    MANMADE = 9

    TRAFFIC_CONE = 10
    BARRIER = 11

    OTHER = 12
    """Miscellaneous classes that do not fit the above."""

    def to_default(self) -> "DefaultLidarSegmentationLabel":
        """Inherited, see superclass."""
        return self
