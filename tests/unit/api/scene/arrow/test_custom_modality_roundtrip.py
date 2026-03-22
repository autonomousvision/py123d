"""Roundtrip tests for CustomModality writer and reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_custom_modality import (
    ArrowCustomModalityReader,
    ArrowCustomModalityWriter,
)
from py123d.datatypes import Timestamp
from py123d.datatypes.custom.custom_modality import CustomModality

from ..conftest import make_custom_modality_metadata


class TestCustomModalityRoundtrip:
    def _write_and_read(self, log_dir, modalities):
        metadata = make_custom_modality_metadata("route")
        writer = ArrowCustomModalityWriter(log_dir=log_dir, metadata=metadata)
        for m in modalities:
            writer.write_modality(m)
        writer.close()
        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def test_dict_data(self, tmp_path: Path):
        metadata = make_custom_modality_metadata("route")
        mod = CustomModality(data={"key": "value", "num": 42}, metadata=metadata, timestamp=Timestamp.from_us(1000))
        table = self._write_and_read(tmp_path, [mod])

        result = ArrowCustomModalityReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.data["key"] == "value"
        assert result.data["num"] == 42
        assert result.timestamp.time_us == 1000

    def test_numpy_data(self, tmp_path: Path):
        metadata = make_custom_modality_metadata("route")
        mod = CustomModality(
            data={"array": np.array([1.0, 2.0, 3.0])},
            metadata=metadata,
            timestamp=Timestamp.from_us(2000),
        )
        table = self._write_and_read(tmp_path, [mod])

        result = ArrowCustomModalityReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        np.testing.assert_array_almost_equal(result.data["array"], [1.0, 2.0, 3.0])

    def test_nested_dict(self, tmp_path: Path):
        metadata = make_custom_modality_metadata("route")
        nested_data = {"level1": {"level2": {"value": 99}}, "list": [1, 2, 3]}
        mod = CustomModality(data=nested_data, metadata=metadata, timestamp=Timestamp.from_us(3000))
        table = self._write_and_read(tmp_path, [mod])

        result = ArrowCustomModalityReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.data["level1"]["level2"]["value"] == 99
        assert result.data["list"] == [1, 2, 3]

    def test_read_column_data_deserialized(self, tmp_path: Path):
        metadata = make_custom_modality_metadata("route")
        mod = CustomModality(data={"x": 1}, metadata=metadata, timestamp=Timestamp.from_us(4000))
        table = self._write_and_read(tmp_path, [mod])

        result = ArrowCustomModalityReader.read_column_at_index(
            0, table, metadata, "data", "test-dataset", deserialize=True
        )
        assert isinstance(result, dict)
        assert result["x"] == 1

    def test_read_column_timestamp_deserialized(self, tmp_path: Path):
        metadata = make_custom_modality_metadata("route")
        mod = CustomModality(data={"x": 1}, metadata=metadata, timestamp=Timestamp.from_us(5000))
        table = self._write_and_read(tmp_path, [mod])

        result = ArrowCustomModalityReader.read_column_at_index(
            0, table, metadata, "timestamp_us", "test-dataset", deserialize=True
        )
        assert isinstance(result, Timestamp)
        assert result.time_us == 5000
