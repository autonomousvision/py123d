from __future__ import annotations

import abc
import importlib
from typing import Dict, Tuple, Type

import numpy as np
import numpy.typing as npt

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


# The canonical Cityscapes display palette [1]_, keyed by the unified camera label. This is the single source of
# truth for camera-segmentation colors: :data:`py123d.visualization.color.default.DEFAULT_CAMERA_SEGMENTATION_COLORS`
# derives its ``Color`` objects from these tuples. It lives in the data layer (as plain RGB tuples, no visualization
# dependency) so that :attr:`py123d.datatypes.Camera.rgb_image` can colorize a semantic label map without importing
# ``py123d.visualization`` (which would invert the package dependency direction).
# [1] https://github.com/mcordts/cityscapesScripts (labels.py)
DEFAULT_CAMERA_SEGMENTATION_RGB: Dict[DefaultCameraSegmentationLabel, Tuple[int, int, int]] = {
    DefaultCameraSegmentationLabel.IGNORE: (0, 0, 0),
    DefaultCameraSegmentationLabel.ROAD: (128, 64, 128),
    DefaultCameraSegmentationLabel.SIDEWALK: (244, 35, 232),
    DefaultCameraSegmentationLabel.BUILDING: (70, 70, 70),
    DefaultCameraSegmentationLabel.POLE: (153, 153, 153),
    DefaultCameraSegmentationLabel.TRAFFIC_LIGHT: (250, 170, 30),
    DefaultCameraSegmentationLabel.TRAFFIC_SIGN: (220, 220, 0),
    DefaultCameraSegmentationLabel.VEGETATION: (107, 142, 35),
    DefaultCameraSegmentationLabel.TERRAIN: (152, 251, 152),
    DefaultCameraSegmentationLabel.SKY: (70, 130, 180),
    DefaultCameraSegmentationLabel.PERSON: (220, 20, 60),
    DefaultCameraSegmentationLabel.RIDER: (255, 0, 0),
    DefaultCameraSegmentationLabel.VEHICLE: (0, 0, 142),
    DefaultCameraSegmentationLabel.TWO_WHEELER: (119, 11, 32),
    DefaultCameraSegmentationLabel.OTHER: (120, 120, 120),
}


def colorize_semantic_label_map(
    label_map: npt.NDArray[np.integer], label_class: Type[CameraSegmentationLabel]
) -> npt.NDArray[np.uint8]:
    """Color a 2D per-pixel class-id map with the Cityscapes palette via the unified default label.

    Each raw, dataset-native class id is mapped to its :class:`DefaultCameraSegmentationLabel`
    (``to_default()``) and then to the canonical Cityscapes-palette color
    (:data:`DEFAULT_CAMERA_SEGMENTATION_RGB`). Ids unknown to the taxonomy fall back to the ``OTHER`` color.
    Mirrors the LiDAR colorizer ``py123d.visualization.matplotlib.lidar._segmentation_colormap`` but operates on a
    2D label map instead of a 1D per-point array.

    :param label_map: ``(H, W)`` array of raw class ids (uint8 or uint16).
    :param label_class: The dataset's :class:`CameraSegmentationLabel` enum for the stored class ids.
    :return: ``(H, W, 3)`` array of RGB uint8 values.
    """
    ids = np.asarray(label_map).astype(np.int64)
    member_ids = [int(member) for member in label_class]
    lut_size = max(int(ids.max()) if ids.size else 0, max(member_ids) if member_ids else 0) + 1

    other_rgb = np.array(DEFAULT_CAMERA_SEGMENTATION_RGB[DefaultCameraSegmentationLabel.OTHER], dtype=np.uint8)
    lut = np.tile(other_rgb, (lut_size, 1))
    for member in label_class:
        rgb = DEFAULT_CAMERA_SEGMENTATION_RGB.get(member.to_default())
        if rgb is not None:
            lut[int(member)] = rgb

    return np.ascontiguousarray(lut[np.clip(ids, 0, lut_size - 1)])
