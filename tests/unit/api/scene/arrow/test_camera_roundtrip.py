"""Roundtrip tests for Camera writer and reader with JPEG/PNG/MP4 codecs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader, ArrowCameraWriter
from py123d.common.io.camera.jpeg_camera_io import is_jpeg_binary
from py123d.common.io.camera.png_camera_io import encode_image_as_png_binary
from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion, PinholeIntrinsics
from py123d.geometry.pose import PoseSE3


def _make_camera_metadata(camera_id: CameraID = CameraID.PCAM_F0) -> PinholeCameraMetadata:
    """Create a minimal PinholeCameraMetadata for testing."""
    return PinholeCameraMetadata(
        camera_name="front_camera",
        camera_id=camera_id,
        intrinsics=PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
        distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        width=640,
        height=480,
        camera_to_imu_se3=PoseSE3.identity(),
    )


def _make_random_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a random RGB uint8 image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def _make_camera(ts_us: int, camera_id: CameraID = CameraID.PCAM_F0) -> Camera:
    metadata = _make_camera_metadata(camera_id)
    return Camera(
        metadata=metadata,
        image=_make_random_image(),
        camera_to_global_se3=PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        timestamp=Timestamp.from_us(ts_us),
    )


class TestCameraJpegBinaryRoundtrip:
    """Write camera data with JPEG binary codec, read back."""

    def _write_and_read(self, log_dir: Path, cameras: list) -> pa.Table:
        metadata = _make_camera_metadata()
        writer = ArrowCameraWriter(log_dir=log_dir, metadata=metadata, camera_codec="jpeg_binary")
        for cam in cameras:
            writer.write_modality(cam)
        writer.close()
        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def test_single_frame(self, tmp_path: Path):
        cam = _make_camera(1000)
        table = self._write_and_read(tmp_path, [cam])
        assert table.num_rows == 1

        metadata = _make_camera_metadata()
        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert isinstance(result, Camera)
        assert result.timestamp.time_us == 1000
        # JPEG is lossy, so just check shape
        assert result.image.shape[:2] == (480, 640)

    def test_multiple_frames(self, tmp_path: Path):
        cameras = [_make_camera(i * 100_000) for i in range(3)]
        table = self._write_and_read(tmp_path, cameras)
        assert table.num_rows == 3

        metadata = _make_camera_metadata()
        for i in range(3):
            result = ArrowCameraReader.read_at_index(i, table, metadata, "test-dataset")
            assert result is not None
            assert result.timestamp.time_us == i * 100_000

    def test_pose_survives_roundtrip(self, tmp_path: Path):
        cam = _make_camera(1000)
        table = self._write_and_read(tmp_path, [cam])

        metadata = _make_camera_metadata()
        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        np.testing.assert_array_almost_equal(result.camera_to_global_se3, cam.camera_to_global_se3, decimal=10)


class TestCameraPngBinaryRoundtrip:
    """Write camera data with PNG binary codec, read back (lossless)."""

    def test_lossless_roundtrip(self, tmp_path: Path):
        metadata = _make_camera_metadata()
        cam = _make_camera(1000)
        writer = ArrowCameraWriter(log_dir=tmp_path, metadata=metadata, camera_codec="png_binary")
        writer.write_modality(cam)
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        # PNG is lossless, exact equality
        np.testing.assert_array_equal(result.image, cam.image)


class TestCameraReadColumn:
    def test_read_column_timestamp(self, tmp_path: Path):
        metadata = _make_camera_metadata()
        cam = _make_camera(5000)
        writer = ArrowCameraWriter(log_dir=tmp_path, metadata=metadata, camera_codec="jpeg_binary")
        writer.write_modality(cam)
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        ts = ArrowCameraReader.read_column_at_index(
            0, table, metadata, "timestamp_us", "test-dataset", deserialize=True
        )
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 5000

    def test_read_column_pose_deserialized(self, tmp_path: Path):
        metadata = _make_camera_metadata()
        cam = _make_camera(5000)
        writer = ArrowCameraWriter(log_dir=tmp_path, metadata=metadata, camera_codec="jpeg_binary")
        writer.write_modality(cam)
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        pose = ArrowCameraReader.read_column_at_index(
            0, table, metadata, "camera_to_global_se3", "test-dataset", deserialize=True
        )
        assert isinstance(pose, PoseSE3)


class TestCameraMp4Roundtrip:
    """Write camera data with MP4 codec, read back."""

    def _write_and_read(self, log_dir: Path, cameras: list) -> pa.Table:
        metadata = _make_camera_metadata()
        writer = ArrowCameraWriter(log_dir=log_dir, metadata=metadata, camera_codec="mp4")
        for cam in cameras:
            writer.write_modality(cam)
        writer.close()
        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def test_single_frame(self, tmp_path: Path):
        cam = _make_camera(1000)
        table = self._write_and_read(tmp_path, [cam])
        assert table.num_rows == 1

        metadata = _make_camera_metadata()
        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset", log_dir=tmp_path)
        assert result is not None
        assert isinstance(result, Camera)
        assert result.timestamp.time_us == 1000
        # MP4 is lossy, so just check shape
        assert result.image.shape == (480, 640, 3)

    def test_multiple_frames(self, tmp_path: Path):
        cameras = [_make_camera(i * 100_000) for i in range(3)]
        table = self._write_and_read(tmp_path, cameras)
        assert table.num_rows == 3

        metadata = _make_camera_metadata()
        for i in range(3):
            result = ArrowCameraReader.read_at_index(i, table, metadata, "test-dataset", log_dir=tmp_path)
            assert result is not None
            assert result.timestamp.time_us == i * 100_000
            assert result.image.shape == (480, 640, 3)

    def test_mp4_file_exists(self, tmp_path: Path):
        cam = _make_camera(1000)
        self._write_and_read(tmp_path, [cam])
        metadata = _make_camera_metadata()
        mp4_path = tmp_path / f"{metadata.modality_key}.mp4"
        assert mp4_path.exists()

    def test_data_column_stores_frame_indices(self, tmp_path: Path):
        cameras = [_make_camera(i * 100_000) for i in range(3)]
        table = self._write_and_read(tmp_path, cameras)
        metadata = _make_camera_metadata()
        data_col = table[f"{metadata.modality_key}.data"].to_pylist()
        assert data_col == [0, 1, 2]

    def test_pose_survives_roundtrip(self, tmp_path: Path):
        cam = _make_camera(1000)
        table = self._write_and_read(tmp_path, [cam])

        metadata = _make_camera_metadata()
        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset", log_dir=tmp_path)
        assert result is not None
        np.testing.assert_array_almost_equal(result.camera_to_global_se3, cam.camera_to_global_se3, decimal=10)


class TestCameraLazyDecode:
    """Cameras built from inline binary keep the bytes and decode on first image access."""

    def _write_and_read(self, log_dir: Path, codec: str, cam: Camera) -> Camera:
        metadata = _make_camera_metadata()
        writer = ArrowCameraWriter(log_dir=log_dir, metadata=metadata, camera_codec=codec)
        writer.write_modality(cam)
        writer.close()
        table = pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()
        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        return result

    def test_reader_returns_undecoded_binary(self, tmp_path: Path):
        result = self._write_and_read(tmp_path, "jpeg_binary", _make_camera(1000))
        assert result.image_binary is not None
        assert is_jpeg_binary(result.image_binary)
        assert result.image.shape[:2] == (480, 640)

    def test_binary_construction_decodes_lossless(self):
        image = _make_random_image()
        cam = Camera(
            metadata=_make_camera_metadata(),
            image_binary=encode_image_as_png_binary(image),
            camera_to_global_se3=PoseSE3.identity(),
            timestamp=Timestamp.from_us(1000),
        )
        np.testing.assert_array_equal(cam.image, image)

    def test_eager_construction_has_no_binary(self):
        assert _make_camera(1000).image_binary is None

    def test_exactly_one_image_source(self):
        image = _make_random_image()
        with pytest.raises(AssertionError):
            Camera(
                metadata=_make_camera_metadata(),
                image=image,
                image_binary=encode_image_as_png_binary(image),
                camera_to_global_se3=PoseSE3.identity(),
                timestamp=Timestamp.from_us(1000),
            )
        with pytest.raises(AssertionError):
            Camera(
                metadata=_make_camera_metadata(),
                camera_to_global_se3=PoseSE3.identity(),
                timestamp=Timestamp.from_us(1000),
            )

    def test_rewrite_passes_jpeg_binary_through(self, tmp_path: Path):
        (tmp_path / "first").mkdir()
        (tmp_path / "second").mkdir()
        first = self._write_and_read(tmp_path / "first", "jpeg_binary", _make_camera(1000))
        rewritten = self._write_and_read(tmp_path / "second", "jpeg_binary", first)
        assert rewritten.image_binary == first.image_binary
