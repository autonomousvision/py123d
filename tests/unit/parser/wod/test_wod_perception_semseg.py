"""Unit tests for WOD-Perception 2D camera panoptic decoding (semantic + packed instance).

Waymo stores one panoptic PNG per labeled image where
``panoptic_label = semantic * panoptic_label_divisor + instance``. The parser splits it into two
pixel-aligned sibling streams: a uint8 semantic class-id map (``// divisor``, ``CAMERA_SEMANTIC``)
and the raw uint16 panoptic map kept verbatim as the packed instance map (``CAMERA_INSTANCE``).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

# WOD parsing needs the optional ``waymo`` extra (tensorflow); skip cleanly when it is absent.
tf = pytest.importorskip("tensorflow")

# The vendored ``*_pb2`` stubs omit their dependency imports, so the protobuf descriptor pool must be
# populated in dependency order before ``dataset_pb2`` (pulled in transitively by the loader) loads.
# An explicit ordered loop is used so an import sorter cannot reshuffle the dependency order.
for _proto in ("vector_pb2", "keypoint_pb2", "map_pb2", "label_pb2", "dataset_pb2"):
    importlib.import_module(f"py123d.parser.wod.waymo_open_dataset.protos.{_proto}")

from py123d.parser.wod.wod_perception_sensor_io import (  # noqa: E402
    load_wod_perception_camera_panoptic_labels,
)

DIVISOR = 1000


def _encode_panoptic_label(panoptic: np.ndarray, divisor: int = DIVISOR) -> SimpleNamespace:
    """Wrap a ``(H, W)`` uint16 panoptic map as a minimal stand-in for ``CameraSegmentationLabel``."""
    png = tf.image.encode_png(panoptic[..., None].astype(np.uint16)).numpy()
    return SimpleNamespace(panoptic_label=png, panoptic_label_divisor=divisor)


def test_panoptic_split_into_semantic_and_packed_instance():
    # (semantic, instance) for a 2x3 image: two TYPE_CAR instances, two TYPE_PEDESTRIAN, background.
    semantic = np.array([[0, 2, 2], [9, 9, 0]], dtype=np.int64)
    instance = np.array([[0, 1, 2], [1, 1, 0]], dtype=np.int64)
    panoptic = (semantic * DIVISOR + instance).astype(np.uint16)

    semantic_out, instance_out = load_wod_perception_camera_panoptic_labels(_encode_panoptic_label(panoptic))

    assert semantic_out is not None and instance_out is not None
    assert semantic_out.dtype == np.uint8
    assert instance_out.dtype == np.uint16
    # Semantic stream is panoptic // divisor; instance stream is the packed panoptic verbatim.
    np.testing.assert_array_equal(semantic_out, semantic.astype(np.uint8))
    np.testing.assert_array_equal(instance_out, panoptic)
    # The instance stream is self-consistent with its semantic sibling.
    np.testing.assert_array_equal(instance_out // DIVISOR, semantic_out)


def test_packed_instance_keeps_distinct_classes_distinct():
    # Same local instance id (1) but different classes must not collide in the packed map.
    semantic = np.array([[2, 9]], dtype=np.int64)  # car vs pedestrian
    instance = np.array([[1, 1]], dtype=np.int64)
    panoptic = (semantic * DIVISOR + instance).astype(np.uint16)

    _, instance_out = load_wod_perception_camera_panoptic_labels(_encode_panoptic_label(panoptic))

    assert instance_out[0, 0] != instance_out[0, 1]


def test_unannotated_frame_returns_none():
    label = SimpleNamespace(panoptic_label=b"", panoptic_label_divisor=DIVISOR)
    semantic_out, instance_out = load_wod_perception_camera_panoptic_labels(label)
    assert semantic_out is None
    assert instance_out is None
