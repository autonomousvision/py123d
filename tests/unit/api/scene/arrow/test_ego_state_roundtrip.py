"""Roundtrip tests for EgoStateSE3 writer and reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import (
    ArrowEgoStateSE3Reader,
    ArrowEgoStateSE3Writer,
)
from py123d.datatypes import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D
from tests.unit.api.scene.conftest import make_ego_metadata


class TestEgoStateSE3Roundtrip:
    """Write ego states via writer, read back via reader, verify correctness."""

    def _write_and_read_table(self, log_dir: Path, ego_states: list) -> pa.Table:
        """Write ego states and return the Arrow table."""
        metadata = make_ego_metadata()
        writer = ArrowEgoStateSE3Writer(log_dir=log_dir, metadata=metadata)
        for ego in ego_states:
            writer.write_modality(ego)
        writer.close()

        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def _make_ego(self, ts_us: int, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> EgoStateSE3:
        metadata = make_ego_metadata()
        pose = PoseSE3(x=x, y=y, z=z, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        dynamic = DynamicStateSE3(
            velocity=Vector3D(1.0, 0.0, 0.0),
            acceleration=Vector3D(0.0, 0.0, 0.0),
            angular_velocity=Vector3D(0.0, 0.0, 0.1),
        )
        return EgoStateSE3.from_imu(
            imu_se3=pose,
            metadata=metadata,
            timestamp=Timestamp.from_us(ts_us),
            dynamic_state_se3=dynamic,
        )

    def test_single_frame(self, tmp_path: Path):
        ego = self._make_ego(1000)
        table = self._write_and_read_table(tmp_path, [ego])
        assert table.num_rows == 1

        metadata = make_ego_metadata()
        result = ArrowEgoStateSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.timestamp.time_us == 1000

    def test_multiple_frames(self, tmp_path: Path):
        egos = [self._make_ego(i * 100) for i in range(10)]
        table = self._write_and_read_table(tmp_path, egos)
        assert table.num_rows == 10

        metadata = make_ego_metadata()
        for i in range(10):
            result = ArrowEgoStateSE3Reader.read_at_index(i, table, metadata, "test-dataset")
            assert result is not None
            assert result.timestamp.time_us == i * 100

    def test_pose_precision(self, tmp_path: Path):
        """Non-trivial pose should survive serialization with exact float equality."""
        metadata = make_ego_metadata()
        pose = PoseSE3(x=1.234567890123, y=-9.876543210987, z=0.001, qw=0.7071, qx=0.0, qy=0.7071, qz=0.0)
        ego = EgoStateSE3.from_imu(imu_se3=pose, metadata=metadata, timestamp=Timestamp.from_us(500))
        table = self._write_and_read_table(tmp_path, [ego])

        result = ArrowEgoStateSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        np.testing.assert_array_almost_equal(result.imu_se3, ego.imu_se3, decimal=10)

    def test_dynamic_state_roundtrip(self, tmp_path: Path):
        """DynamicStateSE3 fields should survive roundtrip."""
        ego = self._make_ego(1000, x=5.0)
        table = self._write_and_read_table(tmp_path, [ego])

        metadata = make_ego_metadata()
        result = ArrowEgoStateSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.dynamic_state_se3 is not None
        np.testing.assert_array_almost_equal(result.dynamic_state_se3, ego.dynamic_state_se3, decimal=10)

    def test_read_column_raw(self, tmp_path: Path):
        """read_column_at_index with deserialize=False returns raw Python value."""
        ego = self._make_ego(1000)
        table = self._write_and_read_table(tmp_path, [ego])

        metadata = make_ego_metadata()
        raw_ts = ArrowEgoStateSE3Reader.read_column_at_index(0, table, metadata, "timestamp_us", "test-dataset")
        assert raw_ts == 1000
        assert isinstance(raw_ts, int)

    def test_read_column_deserialized(self, tmp_path: Path):
        """read_column_at_index with deserialize=True returns domain type."""
        ego = self._make_ego(1000)
        table = self._write_and_read_table(tmp_path, [ego])

        metadata = make_ego_metadata()
        ts = ArrowEgoStateSE3Reader.read_column_at_index(
            0, table, metadata, "timestamp_us", "test-dataset", deserialize=True
        )
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 1000

        pose = ArrowEgoStateSE3Reader.read_column_at_index(
            0, table, metadata, "imu_se3", "test-dataset", deserialize=True
        )
        assert isinstance(pose, PoseSE3)

    def test_read_column_missing_raises(self, tmp_path: Path):
        """Requesting a nonexistent column should raise ValueError."""
        ego = self._make_ego(1000)
        table = self._write_and_read_table(tmp_path, [ego])

        metadata = make_ego_metadata()
        with pytest.raises(ValueError, match="not found in Arrow table"):
            ArrowEgoStateSE3Reader.read_column_at_index(0, table, metadata, "nonexistent_column", "test-dataset")
