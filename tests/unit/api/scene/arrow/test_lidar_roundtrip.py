"""Roundtrip tests for Lidar writer and reader with binary codecs (IPC, Draco, LAZ)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader, ArrowLidarWriter
from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMergedMetadata, LidarMetadata
from py123d.geometry.pose import PoseSE3
from tests.unit.api.scene.conftest import make_log_metadata


def _make_lidar_metadata(lidar_id: LidarID = LidarID.LIDAR_TOP) -> LidarMetadata:
    return LidarMetadata(
        lidar_name="top_lidar",
        lidar_id=lidar_id,
        lidar_to_imu_se3=PoseSE3.identity(),
    )


def _make_merged_metadata() -> LidarMergedMetadata:
    top = LidarMetadata(lidar_name="top_lidar", lidar_id=LidarID.LIDAR_TOP)
    return LidarMergedMetadata(lidar_metadata_dict={LidarID.LIDAR_TOP: top})


def _make_point_cloud(num_points: int = 100) -> tuple:
    """Create a random point cloud with xyz + features."""
    rng = np.random.RandomState(42)
    xyz = rng.randn(num_points, 3).astype(np.float32)
    features = {
        LidarFeature.INTENSITY.serialize(): rng.randint(0, 255, size=num_points).astype(np.float32),
    }
    return xyz, features


def _make_lidar(ts_us: int, metadata=None, num_points: int = 100) -> Lidar:
    if metadata is None:
        metadata = _make_merged_metadata()
    xyz, features = _make_point_cloud(num_points)
    return Lidar(
        timestamp=Timestamp.from_us(ts_us),
        timestamp_end=Timestamp.from_us(ts_us + 50_000),
        metadata=metadata,
        point_cloud_3d=xyz,
        point_cloud_features=features,
    )


class TestLidarIpcBinaryRoundtrip:
    """Write lidar data with IPC binary codec, read back."""

    def _write_and_read(self, log_dir: Path, lidars: list, codec: str = "ipc") -> pa.Table:
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=log_dir,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec=codec,
        )
        for lidar in lidars:
            writer.write_modality(lidar)
        writer.close()
        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def test_single_frame_ipc(self, tmp_path: Path):
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc")
        assert table.num_rows == 1

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert isinstance(result, Lidar)
        assert result.timestamp.time_us == 1000
        assert result.point_cloud_3d.shape == (100, 3)

    def test_ipc_zstd_codec(self, tmp_path: Path):
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc_zstd")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape == (100, 3)

    def test_ipc_lz4_codec(self, tmp_path: Path):
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc_lz4")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape == (100, 3)

    def test_point_cloud_data_preserved(self, tmp_path: Path):
        """Verify xyz coordinates survive IPC roundtrip (lossless)."""
        lidar = _make_lidar(1000)
        original_xyz = lidar.point_cloud_3d.copy()
        table = self._write_and_read(tmp_path, [lidar], codec="ipc")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        np.testing.assert_array_almost_equal(result.point_cloud_3d, original_xyz, decimal=5)

    def test_multiple_frames(self, tmp_path: Path):
        lidars = [_make_lidar(i * 100_000) for i in range(5)]
        table = self._write_and_read(tmp_path, lidars, codec="ipc")
        assert table.num_rows == 5

        metadata = _make_merged_metadata()
        for i in range(5):
            result = ArrowLidarReader.read_at_index(i, table, metadata, "test-dataset")
            assert result is not None
            assert result.timestamp.time_us == i * 100_000

    def test_timestamps_roundtrip(self, tmp_path: Path):
        """Verify start and end timestamps survive roundtrip."""
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.timestamp.time_us == 1000
        assert result.timestamp_end.time_us == 51_000


class TestLidarDracoRoundtrip:
    """Write lidar data with Draco codec, read back."""

    def test_draco_roundtrip(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="draco",
        )
        lidar = _make_lidar(1000)
        writer.write_modality(lidar)
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape == (100, 3)
        # Draco is lossy for coordinates, just check approximate shape
        assert result.point_cloud_3d.dtype == np.float32


class TestLidarLazRoundtrip:
    """Write lidar data with LAZ codec, read back."""

    def test_laz_roundtrip(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="laz",
        )
        lidar = _make_lidar(1000)
        writer.write_modality(lidar)
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape[1] == 3
        assert result.point_cloud_3d.dtype == np.float32


class TestLidarReadColumn:
    def test_read_column_timestamp_deserialized(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="ipc",
        )
        writer.write_modality(_make_lidar(5000))
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        ts = ArrowLidarReader.read_column_at_index(0, table, metadata, "timestamp_us", "test-dataset", deserialize=True)
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 5000

    def test_read_column_end_timestamp(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="ipc",
        )
        writer.write_modality(_make_lidar(5000))
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        ts = ArrowLidarReader.read_column_at_index(
            0, table, metadata, "end_timestamp_us", "test-dataset", deserialize=True
        )
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 55_000  # 5000 + 50_000
