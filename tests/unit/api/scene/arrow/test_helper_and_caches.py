"""Tests for helper.py, arrow_scene_caches.py, arrow_sync.py, and modalities/utils.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

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
