"""Tests for helper.py, arrow_scene_caches.py, arrow_sync.py, and modalities/utils.py."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.helper import get_scene_anchor_timestamps
from py123d.api.scene.arrow.modalities.arrow_sync import get_timestamp_from_arrow_table
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema, get_optional_array_mixin
from py123d.api.scene.arrow.utils.arrow_scene_caches import _get_complete_log_scene_metadata
from py123d.datatypes import Timestamp
from py123d.geometry.vector import Vector3D

from ..conftest import make_ego_metadata, make_log_metadata, write_sync_arrow

# ===========================================================================
# _get_complete_log_scene_metadata
# ===========================================================================


class TestGetCompleteLogSceneMetadata:
    def test_basic(self, tmp_path: Path):
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        num_rows = 10
        timestep_us = 100_000

        write_sync_arrow(
            tmp_path,
            num_rows,
            timestep_us,
            log_meta,
            modality_columns={ego_meta.modality_key: list(range(num_rows))},
        )

        scene_meta = _get_complete_log_scene_metadata(tmp_path, log_meta)
        assert scene_meta.num_future_iterations == 9  # 10 - 1
        assert scene_meta.num_history_iterations == 0
        assert scene_meta.initial_idx == 0
        assert scene_meta.dataset == "test-dataset"
        assert scene_meta.target_iteration_stride == 1

    def test_single_row(self, tmp_path: Path):
        """Single row: num_future=0, duration=0.0."""
        log_meta = make_log_metadata()
        write_sync_arrow(tmp_path, num_rows=1, timestep_us=100_000, log_metadata=log_meta)

        scene_meta = _get_complete_log_scene_metadata(tmp_path, log_meta)
        assert scene_meta.num_future_iterations == 0
        assert scene_meta.iteration_duration_s == 0.0

    def test_two_rows(self, tmp_path: Path):
        log_meta = make_log_metadata()
        write_sync_arrow(tmp_path, num_rows=2, timestep_us=100_000, log_metadata=log_meta)

        scene_meta = _get_complete_log_scene_metadata(tmp_path, log_meta)
        assert scene_meta.num_future_iterations == 1
        assert abs(scene_meta.iteration_duration_s - 0.1) < 1e-6


# ===========================================================================
# get_timestamp_from_arrow_table
# ===========================================================================


class TestGetTimestampFromArrowTable:
    def test_valid_index(self, tmp_path: Path):
        log_meta = make_log_metadata()
        sync_table = write_sync_arrow(tmp_path, num_rows=10, timestep_us=100_000, log_metadata=log_meta)

        ts = get_timestamp_from_arrow_table(sync_table, 5)
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 5 * 100_000


# ===========================================================================
# modalities/utils.py
# ===========================================================================


class TestGetOptionalArrayMixin:
    def test_none_returns_none(self):
        assert get_optional_array_mixin(None, Vector3D) is None

    def test_list_returns_mixin(self):
        result = get_optional_array_mixin([1.0, 2.0, 3.0], Vector3D)
        assert isinstance(result, Vector3D)
        assert result[0] == 1.0

    def test_ndarray_returns_mixin(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = get_optional_array_mixin(arr, Vector3D)
        assert isinstance(result, Vector3D)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported data type"):
            get_optional_array_mixin("not_supported", Vector3D)


class TestAllColumnsInSchema:
    def test_all_present(self):
        schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
        table = pa.table({"a": [1], "b": ["x"]}, schema=schema)
        assert all_columns_in_schema(table, ["a", "b"]) is True

    def test_one_missing(self):
        schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
        table = pa.table({"a": [1], "b": ["x"]}, schema=schema)
        assert all_columns_in_schema(table, ["a", "c"]) is False

    def test_empty_columns(self):
        schema = pa.schema([("a", pa.int64())])
        table = pa.table({"a": [1]}, schema=schema)
        assert all_columns_in_schema(table, []) is True


# ===========================================================================
# get_scene_anchor_timestamps
# ===========================================================================


class TestGetSceneAnchorTimestamps:
    def _make_scenes(self, tmp_path: Path) -> list:
        """Two logs with different timesteps, three scenes each at distinct anchor indices."""
        scenes = []
        for log_index, log_name in enumerate(["log_001", "log_002"]):
            log_dir = tmp_path / "test-dataset_train" / log_name
            log_dir.mkdir(parents=True)
            log_meta = make_log_metadata(log_name=log_name)
            write_sync_arrow(log_dir, num_rows=10, timestep_us=100_000 * (log_index + 1), log_metadata=log_meta)
            base_meta = _get_complete_log_scene_metadata(log_dir, log_meta)
            for initial_idx in (0, 3, 7):
                scene_meta = replace(base_meta, initial_idx=initial_idx, num_future_iterations=9 - initial_idx)
                scenes.append(ArrowSceneAPI(log_dir, scene_meta))
        return scenes

    def test_matches_per_scene_reads(self, tmp_path: Path):
        scenes = self._make_scenes(tmp_path)
        bulk = get_scene_anchor_timestamps(scenes)
        per_scene = [scene.get_timestamp_at_iteration(0) for scene in scenes]
        assert [t.time_us for t in bulk] == [t.time_us for t in per_scene]

    def test_empty(self):
        assert get_scene_anchor_timestamps([]) == []

    def test_non_arrow_scene_falls_back_to_per_scene_read(self):
        class _StubScene:
            def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
                assert iteration == 0
                return Timestamp.from_us(42)

        result = get_scene_anchor_timestamps([_StubScene()])
        assert [t.time_us for t in result] == [42]

    def test_reads_each_log_sync_table_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from py123d.api.scene.arrow import helper

        scenes = self._make_scenes(tmp_path)
        sync_table_reads: list[Path] = []
        real_get_sync_table = helper.get_sync_table

        def _counting_get_sync_table(log_dir: Path) -> pa.Table:
            sync_table_reads.append(log_dir)
            return real_get_sync_table(log_dir)

        monkeypatch.setattr(helper, "get_sync_table", _counting_get_sync_table)
        get_scene_anchor_timestamps(scenes)
        assert len(sync_table_reads) == 2  # six scenes across two logs
