"""Unit tests for PandaSet per-point semantic segmentation loading (sparse, per-log)."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py123d.parser.pandaset.pandaset_sensor_io import _load_pandaset_semseg


def _write_semseg(log_path: Path, iteration: int, classes: np.ndarray) -> None:
    """Write a synthetic PandaSet ``annotations/semseg/{iteration:02d}.pkl.gz`` (single ``class`` column)."""
    semseg_dir = log_path / "annotations" / "semseg"
    semseg_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"class": classes})
    with gzip.open(semseg_dir / f"{iteration:02d}.pkl.gz", "wb") as f:
        pickle.dump(df, f)


class TestLoadPandasetSemseg:
    def test_returns_none_when_unannotated(self, tmp_path: Path):
        # A log/frame without a semseg file is the common case (only a subset of logs is annotated).
        assert _load_pandaset_semseg(tmp_path, iteration=0, num_points=100) is None

    def test_loads_aligned_uint8_labels(self, tmp_path: Path):
        rng = np.random.RandomState(0)
        classes = rng.randint(1, 43, size=128).astype(np.int64)  # raw PandaSet ids are 1..42
        _write_semseg(tmp_path, iteration=3, classes=classes)

        semantic = _load_pandaset_semseg(tmp_path, iteration=3, num_points=128)
        assert semantic is not None
        assert semantic.dtype == np.uint8
        assert semantic.shape == (128,)
        np.testing.assert_array_equal(semantic, classes.astype(np.uint8))

    def test_raises_on_misalignment(self, tmp_path: Path):
        _write_semseg(tmp_path, iteration=0, classes=np.ones(64, dtype=np.int64))
        with pytest.raises(AssertionError, match="misaligned"):
            _load_pandaset_semseg(tmp_path, iteration=0, num_points=65)


class TestPandasetLidarMetadataTaxonomy:
    """The merged lidar metadata must tag its semseg taxonomy on the primary sensor and round-trip it."""

    def test_taxonomy_tagged_on_primary_lidar(self):
        from py123d.datatypes.sensors.lidar import LidarID
        from py123d.parser.lidar_segmentation_registry import PandasetLidarSegmentationLabel
        from py123d.parser.pandaset.utils.pandaset_constants import PANDASET_LIDAR_MERGED_METADATA

        # The primary (LIDAR_TOP / main_pandar64) carries the taxonomy; the secondary lidar does not.
        assert PANDASET_LIDAR_MERGED_METADATA[LidarID.LIDAR_TOP].segmentation_label_class is (
            PandasetLidarSegmentationLabel
        )
        assert PANDASET_LIDAR_MERGED_METADATA[LidarID.LIDAR_FRONT].segmentation_label_class is None

    def test_taxonomy_survives_metadata_roundtrip(self):
        from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata
        from py123d.parser.lidar_segmentation_registry import PandasetLidarSegmentationLabel
        from py123d.parser.pandaset.utils.pandaset_constants import PANDASET_LIDAR_MERGED_METADATA

        top = PANDASET_LIDAR_MERGED_METADATA[LidarID.LIDAR_TOP]
        restored = LidarMetadata.from_dict(top.to_dict())
        assert restored.segmentation_label_class is PandasetLidarSegmentationLabel
