"""Tests for SceneFilter validation — deduplication, map, stride, chunking, modality requirements."""

from __future__ import annotations

import logging

import pytest

from py123d.api.scene.scene_filter import SceneFilter, _validate_modality_requirement


class TestDeduplication:
    def test_datasets_deduplicated(self):
        f = SceneFilter(datasets=["a", "b", "a"])
        assert f.datasets == ["a", "b"]

    def test_split_types_deduplicated(self):
        f = SceneFilter(split_types=["train", "val", "train"])
        assert f.split_types == ["train", "val"]

    def test_log_names_deduplicated(self):
        f = SceneFilter(log_names=["log1", "log2", "log1"])
        assert f.log_names == ["log1", "log2"]

    def test_order_preserved(self):
        f = SceneFilter(datasets=["b", "a", "b"])
        assert f.datasets == ["b", "a"]

    def test_none_stays_none(self):
        f = SceneFilter(datasets=None)
        assert f.datasets is None

    def test_map_locations_deduplicated(self):
        f = SceneFilter(map_locations=["boston", "boston", "palo_alto"])
        assert f.map_locations == ["boston", "palo_alto"]

    def test_required_modalities_deduplicated(self):
        f = SceneFilter(required_scene_modalities=["ego_state_se3", "ego_state_se3", "camera:any"])
        assert f.required_scene_modalities == ["ego_state_se3", "camera:any"]


class TestMapValidation:
    def test_map_has_z_with_has_map_false_raises(self):
        with pytest.raises(ValueError, match="Cannot filter by map elevation"):
            SceneFilter(has_map=False, map_has_z=True)

    def test_map_has_z_with_has_map_none_ok(self):
        f = SceneFilter(has_map=None, map_has_z=True)
        assert f.map_has_z is True

    def test_map_has_z_with_has_map_true_ok(self):
        f = SceneFilter(has_map=True, map_has_z=True)
        assert f.map_has_z is True


class TestStrideValidation:
    def test_stride_zero_raises(self):
        with pytest.raises(ValueError, match="target_iteration_stride must be >= 1"):
            SceneFilter(target_iteration_stride=0)

    def test_stride_negative_raises(self):
        with pytest.raises(ValueError, match="target_iteration_stride must be >= 1"):
            SceneFilter(target_iteration_stride=-1)

    def test_duration_zero_raises(self):
        with pytest.raises(ValueError, match="target_iteration_duration_s must be > 0"):
            SceneFilter(target_iteration_duration_s=0.0)

    def test_duration_negative_raises(self):
        with pytest.raises(ValueError, match="target_iteration_duration_s must be > 0"):
            SceneFilter(target_iteration_duration_s=-0.5)

    def test_both_stride_and_duration_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            SceneFilter(target_iteration_stride=2, target_iteration_duration_s=0.5)
        assert "target_iteration_duration_s takes priority" in caplog.text


class TestDurationConflicts:
    def test_future_both_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            SceneFilter(future_duration_s=5.0, future_num_iterations=10)
        assert "future_duration_s" in caplog.text

    def test_history_both_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            SceneFilter(history_duration_s=2.0, history_num_iterations=5)
        assert "history_duration_s" in caplog.text

    def test_threshold_both_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            SceneFilter(timestamp_threshold_s=1.0, iteration_threshold=10)
        assert "timestamp_threshold_s takes priority" in caplog.text


class TestModalityRequirements:
    def test_valid_exact_key(self):
        _validate_modality_requirement("camera.pcam_f0")

    def test_valid_any_pattern(self):
        _validate_modality_requirement("camera:any")

    def test_valid_all_pattern(self):
        _validate_modality_requirement("camera:all")

    def test_invalid_quantifier_raises(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            _validate_modality_requirement("camera:some")

    def test_multiple_colons_raises(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            _validate_modality_requirement("a:b:c")

    def test_modality_requirements_validated_on_init(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            SceneFilter(required_scene_modalities=["camera:invalid"])


class TestChunkingValidation:
    def test_num_without_idx_raises(self):
        with pytest.raises(ValueError, match="Both num_chunks and chunk_idx must be set together"):
            SceneFilter(num_chunks=4, chunk_idx=None)

    def test_idx_without_num_raises(self):
        with pytest.raises(ValueError, match="Both num_chunks and chunk_idx must be set together"):
            SceneFilter(num_chunks=None, chunk_idx=2)

    def test_idx_gte_num_raises(self):
        with pytest.raises(ValueError, match="chunk_idx.*must be < num_chunks"):
            SceneFilter(num_chunks=4, chunk_idx=4)

    def test_idx_eq_num_minus_1_ok(self):
        f = SceneFilter(num_chunks=4, chunk_idx=3)
        assert f.num_chunks == 4

    def test_valid_chunking(self):
        f = SceneFilter(num_chunks=4, chunk_idx=0)
        assert f.chunk_idx == 0


class TestSceneUUIDNormalization:
    def test_uuids_deduplicated(self):
        uid = "00000000-0000-0000-0000-000000000001"
        f = SceneFilter(scene_uuids=[uid, uid])
        assert len(f.scene_uuids) == 1
