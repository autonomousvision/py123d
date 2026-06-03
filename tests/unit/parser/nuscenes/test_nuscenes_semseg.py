"""Unit tests for nuScenes per-point segmentation: SEMANTIC from lidarseg, INSTANCE from panoptic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from py123d.datatypes import LidarFeature
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.lidar_segmentation_registry import NuScenesLidarSegmentationLabel
from py123d.parser.nuscenes.nuscenes_sensor_io import (
    NUSCENES_PANOPTIC_DIVISOR,
    _load_nuscenes_lidarseg,
    _load_nuscenes_panoptic_instance,
    load_nuscenes_point_cloud_data_from_path,
)
from py123d.parser.nuscenes.utils.nuscenes_extraction import (
    _NUSCENES_LIDARSEG_IDX2NAME,
    _validate_nuscenes_lidarseg_taxonomy,
)


def _write_lidarseg(path: Path, semantic: np.ndarray) -> None:
    """Write a synthetic nuScenes ``_lidarseg.bin`` (one uint8 class id per point)."""
    semantic.astype(np.uint8).tofile(path)


def _write_panoptic(path: Path, category: np.ndarray, instance: np.ndarray) -> None:
    """Write a synthetic nuScenes ``_panoptic.npz`` (key ``data``) with the official encoding."""
    panoptic = (category.astype(np.int64) * NUSCENES_PANOPTIC_DIVISOR + instance.astype(np.int64)).astype(np.int32)
    np.savez(path, data=panoptic)


class TestLoadNuScenesLidarseg:
    def test_returns_none_when_no_path(self):
        assert _load_nuscenes_lidarseg(None, num_points=100) is None

    def test_returns_none_when_file_missing(self, tmp_path: Path):
        assert _load_nuscenes_lidarseg(tmp_path / "missing_lidarseg.bin", num_points=100) is None

    def test_loads_aligned_uint8_labels(self, tmp_path: Path):
        rng = np.random.RandomState(0)
        classes = rng.randint(0, 32, size=128)  # raw nuScenes lidarseg ids are 0..31
        path = tmp_path / "abc_lidarseg.bin"
        _write_lidarseg(path, classes)

        semantic = _load_nuscenes_lidarseg(path, num_points=128)
        assert semantic is not None
        assert semantic.dtype == np.uint8
        assert semantic.shape == (128,)
        np.testing.assert_array_equal(semantic, classes.astype(np.uint8))

    def test_raises_on_misalignment(self, tmp_path: Path):
        path = tmp_path / "x_lidarseg.bin"
        _write_lidarseg(path, np.ones(64, dtype=np.uint8))
        with pytest.raises(AssertionError, match="misaligned"):
            _load_nuscenes_lidarseg(path, num_points=65)


class TestLoadNuScenesPanopticInstance:
    def test_returns_none_when_no_path(self):
        assert _load_nuscenes_panoptic_instance(None, num_points=100) is None

    def test_decodes_instance_remainder(self, tmp_path: Path):
        rng = np.random.RandomState(1)
        category = rng.randint(0, 32, size=128)
        instance = rng.randint(0, NUSCENES_PANOPTIC_DIVISOR, size=128)  # 0 == stuff / un-instanced
        path = tmp_path / "abc_panoptic.npz"
        _write_panoptic(path, category, instance)

        instance_out = _load_nuscenes_panoptic_instance(path, num_points=128)
        assert instance_out is not None
        assert instance_out.dtype == np.uint16
        assert instance_out.shape == (128,)
        np.testing.assert_array_equal(instance_out, instance.astype(np.uint16))

    def test_raises_on_misalignment(self, tmp_path: Path):
        path = tmp_path / "x_panoptic.npz"
        _write_panoptic(path, np.ones(64, dtype=np.int64), np.zeros(64, dtype=np.int64))
        with pytest.raises(AssertionError, match="misaligned"):
            _load_nuscenes_panoptic_instance(path, num_points=65)


class TestLoadNuScenesPointCloudSegmentation:
    """The top-level loader must take SEMANTIC from lidarseg and INSTANCE from panoptic, not panoptic//1000."""

    def _write_pcd(self, path: Path, num_points: int) -> None:
        rng = np.random.RandomState(2)
        pcd = rng.randn(num_points, 5).astype(np.float32)  # x, y, z, intensity, ring
        pcd.tofile(path)

    def test_semantic_from_lidarseg_not_panoptic(self, tmp_path: Path):
        num_points = 64
        pcd_path = tmp_path / "x.pcd.bin"
        self._write_pcd(pcd_path, num_points)

        # lidarseg labels everything CAR (17); panoptic demotes the first half to noise (category 0),
        # which is exactly the thing-class-to-noise drop we must NOT inherit by using panoptic // 1000.
        semantic_gt = np.full(num_points, int(NuScenesLidarSegmentationLabel.CAR), dtype=np.uint8)
        _write_lidarseg(tmp_path / "x_lidarseg.bin", semantic_gt)
        category = semantic_gt.astype(np.int64).copy()
        category[: num_points // 2] = 0  # demoted to noise in panoptic
        instance = np.arange(num_points, dtype=np.int64) % NUSCENES_PANOPTIC_DIVISOR
        _write_panoptic(tmp_path / "x_panoptic.npz", category, instance)

        lidar_metadatas = {LidarID.LIDAR_TOP: LidarMetadata("LIDAR_TOP", LidarID.LIDAR_TOP, PoseSE3.identity())}
        _, features = load_nuscenes_point_cloud_data_from_path(
            pcd_path,
            lidar_metadatas,
            lidarseg_path=tmp_path / "x_lidarseg.bin",
            panoptic_path=tmp_path / "x_panoptic.npz",
        )

        semantic = features[LidarFeature.SEMANTIC.serialize()]
        instance_out = features[LidarFeature.INSTANCE.serialize()]
        # Semantic must be the lossless lidarseg CAR everywhere — never the panoptic-demoted noise.
        np.testing.assert_array_equal(semantic, semantic_gt)
        np.testing.assert_array_equal(instance_out, instance.astype(np.uint16))

    def test_features_absent_when_no_segmentation(self, tmp_path: Path):
        num_points = 32
        pcd_path = tmp_path / "x.pcd.bin"
        self._write_pcd(pcd_path, num_points)
        lidar_metadatas = {LidarID.LIDAR_TOP: LidarMetadata("LIDAR_TOP", LidarID.LIDAR_TOP, PoseSE3.identity())}
        _, features = load_nuscenes_point_cloud_data_from_path(pcd_path, lidar_metadatas)
        assert LidarFeature.SEMANTIC.serialize() not in features
        assert LidarFeature.INSTANCE.serialize() not in features


class TestNuScenesTaxonomyGuard:
    """The conversion-time guard must accept the canonical indexing and reject any drift."""

    def test_accepts_matching_mapping(self):
        nusc = SimpleNamespace(lidarseg=[{}], lidarseg_idx2name_mapping=dict(_NUSCENES_LIDARSEG_IDX2NAME))
        _validate_nuscenes_lidarseg_taxonomy(nusc)  # must not raise

    def test_rejects_drifted_mapping(self):
        drifted = dict(_NUSCENES_LIDARSEG_IDX2NAME)
        drifted[17] = "vehicle.truck"  # car/truck swapped on disk → must fail loudly
        nusc = SimpleNamespace(lidarseg=[{}], lidarseg_idx2name_mapping=drifted)
        with pytest.raises(AssertionError, match="taxonomy mismatch"):
            _validate_nuscenes_lidarseg_taxonomy(nusc)

    def test_rejects_missing_mapping(self):
        nusc = SimpleNamespace(lidarseg=[{}], lidarseg_idx2name_mapping=None)
        with pytest.raises(AssertionError, match="lidarseg_idx2name_mapping"):
            _validate_nuscenes_lidarseg_taxonomy(nusc)


class TestNuScenesLidarMetadataTaxonomy:
    """The top-lidar metadata must carry and round-trip the nuScenes segmentation taxonomy."""

    def test_taxonomy_survives_metadata_roundtrip(self):
        top = LidarMetadata(
            lidar_name="LIDAR_TOP",
            lidar_id=LidarID.LIDAR_TOP,
            segmentation_label_class=NuScenesLidarSegmentationLabel,
        )
        restored = LidarMetadata.from_dict(top.to_dict())
        assert restored.segmentation_label_class is NuScenesLidarSegmentationLabel
        # A merged metadata wrapping it must expose the same taxonomy on the top lidar.
        merged = LidarMergedMetadata({LidarID.LIDAR_TOP: top})
        assert merged[LidarID.LIDAR_TOP].segmentation_label_class is NuScenesLidarSegmentationLabel
