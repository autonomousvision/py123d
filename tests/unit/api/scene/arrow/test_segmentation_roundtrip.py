"""Roundtrip tests for semantic segmentation: per-point LiDAR labels and per-pixel camera label maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader, ArrowCameraWriter
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader, ArrowLidarWriter, _read_load_kwargs
from py123d.api.utils.arrow_metadata_utils import resolve_metadata_class
from py123d.datatypes import Timestamp
from py123d.datatypes.modalities.base_modality import ModalityType
from py123d.datatypes.sensors.base_camera import Camera, CameraChannelType, CameraID
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion, PinholeIntrinsics
from py123d.datatypes.sensors.segmentation_camera import SegmentationCameraMetadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.base_dataset_parser import ParsedLidar
from py123d.parser.camera_segmentation_registry import WODPerceptionCameraSegmentationLabel
from py123d.parser.lidar_segmentation_registry import (
    NuScenesLidarSegmentationLabel,
    PandasetLidarSegmentationLabel,
    WODPerceptionLidarSegmentationLabel,
)

from ..conftest import make_log_metadata

# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def _make_rgb_camera_metadata(camera_id: CameraID = CameraID.PCAM_F0) -> PinholeCameraMetadata:
    return PinholeCameraMetadata(
        camera_name="front_camera",
        camera_id=camera_id,
        intrinsics=PinholeIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=120.0),
        distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        width=320,
        height=240,
        camera_to_imu_se3=PoseSE3.identity(),
    )


def _make_segmentation_camera_metadata(camera_id: CameraID = CameraID.PCAM_F0) -> SegmentationCameraMetadata:
    return SegmentationCameraMetadata(
        camera_metadata=_make_rgb_camera_metadata(camera_id),
        segmentation_label_class=WODPerceptionCameraSegmentationLabel,
    )


def _make_label_map(dtype=np.uint8, height: int = 240, width: int = 320) -> np.ndarray:
    rng = np.random.RandomState(7)
    high = 28 if dtype == np.uint8 else 4000
    return rng.randint(0, high, size=(height, width)).astype(dtype)


# ----------------------------------------------------------------------------------------------------------------------
# Per-point LiDAR semantic segmentation
# ----------------------------------------------------------------------------------------------------------------------


class TestLidarSemanticFeatureRoundtrip:
    """SEMANTIC / INSTANCE per-point features must survive the lidar codec losslessly."""

    def _write_and_read(self, log_dir: Path, lidar: Lidar, codec: str = "ipc") -> Lidar:
        metadata = lidar.metadata
        writer = ArrowLidarWriter(
            log_dir=log_dir,
            metadata=metadata,
            log_metadata=make_log_metadata(),
            lidar_store_option="binary",
            lidar_codec=codec,
        )
        writer.write_modality(lidar)
        writer.close()
        table = pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        return result

    def _make_lidar_with_semantic(self, num_points: int = 64) -> Lidar:
        rng = np.random.RandomState(3)
        xyz = rng.randn(num_points, 3).astype(np.float32)
        features = {
            LidarFeature.SEMANTIC.serialize(): rng.randint(0, 23, size=num_points).astype(np.uint8),
            LidarFeature.INSTANCE.serialize(): rng.randint(0, 5000, size=num_points).astype(np.uint16),
        }
        return Lidar(
            timestamp=Timestamp.from_us(1000),
            timestamp_end=Timestamp.from_us(51000),
            metadata=LidarMergedMetadata({LidarID.LIDAR_TOP: LidarMetadata("top", LidarID.LIDAR_TOP)}),
            point_cloud_3d=xyz,
            point_cloud_features=features,
        )

    def test_semantic_and_instance_roundtrip_exact(self, tmp_path: Path):
        lidar = self._make_lidar_with_semantic()
        result = self._write_and_read(tmp_path, lidar)

        np.testing.assert_array_equal(result.semantic, lidar.semantic)
        np.testing.assert_array_equal(result.instance, lidar.instance)
        assert result.semantic.dtype == np.uint8
        assert result.instance.dtype == np.uint16

    def test_accessor_returns_none_when_absent(self, tmp_path: Path):
        # A lidar without segmentation features must read back with no semantic/instance (sparsity).
        rng = np.random.RandomState(4)
        lidar = Lidar(
            timestamp=Timestamp.from_us(1000),
            timestamp_end=Timestamp.from_us(51000),
            metadata=LidarMergedMetadata({LidarID.LIDAR_TOP: LidarMetadata("top", LidarID.LIDAR_TOP)}),
            point_cloud_3d=rng.randn(32, 3).astype(np.float32),
            point_cloud_features={LidarFeature.INTENSITY.serialize(): rng.randint(0, 255, 32).astype(np.uint8)},
        )
        result = self._write_and_read(tmp_path, lidar)
        assert result.semantic is None
        assert result.instance is None


# ----------------------------------------------------------------------------------------------------------------------
# Per-pixel camera semantic segmentation
# ----------------------------------------------------------------------------------------------------------------------


class TestSegmentationCameraMetadata:
    """The dedicated segmentation metadata must report its own modality and round-trip the taxonomy."""

    def test_modality_routing(self):
        meta = _make_segmentation_camera_metadata()
        assert meta.modality_type == ModalityType.CAMERA_SEGMENTATION
        assert meta.channel_type == CameraChannelType.SEMANTIC
        assert meta.modality_key == "camera_segmentation.pcam_f0"
        # Geometry is delegated to the sibling RGB camera.
        assert meta.width == 320 and meta.height == 240
        assert meta.camera_id == CameraID.PCAM_F0

    def test_metadata_dict_roundtrip(self):
        meta = _make_segmentation_camera_metadata()
        restored = SegmentationCameraMetadata.from_dict(meta.to_dict())
        assert restored.modality_key == meta.modality_key
        assert restored.segmentation_label_class is WODPerceptionCameraSegmentationLabel
        assert restored.width == meta.width

    def test_reader_path_resolves_segmentation_metadata(self):
        # The log-directory parser resolves the metadata class purely from the modality key.
        meta = _make_segmentation_camera_metadata()
        metadata_class = resolve_metadata_class("camera_segmentation.pcam_f0")
        assert metadata_class is SegmentationCameraMetadata
        restored = metadata_class.from_dict(meta.to_dict())
        assert restored.segmentation_label_class is WODPerceptionCameraSegmentationLabel


class TestSegmentationCameraRoundtrip:
    """Label maps must survive the label_png codec losslessly, with pose/timestamp intact."""

    def _write_and_read(self, log_dir: Path, camera: Camera) -> Camera:
        metadata = camera.metadata
        writer = ArrowCameraWriter(log_dir=log_dir, metadata=metadata, camera_codec="label_png")
        writer.write_modality(camera)
        writer.close()
        table = pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()
        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        return result

    def test_uint8_label_map_lossless(self, tmp_path: Path):
        label_map = _make_label_map(np.uint8)
        camera = Camera(
            metadata=_make_segmentation_camera_metadata(),
            image=label_map,
            camera_to_global_se3=PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            timestamp=Timestamp.from_us(1234),
        )
        result = self._write_and_read(tmp_path, camera)
        assert result.timestamp.time_us == 1234
        assert result.image.shape == label_map.shape
        np.testing.assert_array_equal(result.image, label_map)
        np.testing.assert_array_almost_equal(result.camera_to_global_se3, camera.camera_to_global_se3, decimal=10)

    def test_uint16_label_map_lossless(self, tmp_path: Path):
        label_map = _make_label_map(np.uint16)
        camera = Camera(
            metadata=_make_segmentation_camera_metadata(),
            image=label_map,
            camera_to_global_se3=PoseSE3.identity(),
            timestamp=Timestamp.from_us(1),
        )
        result = self._write_and_read(tmp_path, camera)
        np.testing.assert_array_equal(result.image, label_map)

    def test_segmentation_file_is_separate_from_rgb(self, tmp_path: Path):
        # Writing the RGB and segmentation siblings of the same camera must not collide.
        rgb_meta = _make_rgb_camera_metadata()
        seg_meta = _make_segmentation_camera_metadata()
        assert rgb_meta.modality_key == "camera.pcam_f0"
        assert seg_meta.modality_key == "camera_segmentation.pcam_f0"

        ArrowCameraWriter(log_dir=tmp_path, metadata=seg_meta, camera_codec="label_png").close()
        # The taxonomy is recoverable from the written file's schema metadata alone.
        table = pa.ipc.open_file(str(tmp_path / f"{seg_meta.modality_key}.arrow")).read_all()
        restored = SegmentationCameraMetadata.from_dict(
            __import__("msgpack").unpackb(table.schema.metadata[b"metadata"], raw=False)
        )
        assert restored.segmentation_label_class is WODPerceptionCameraSegmentationLabel


# ----------------------------------------------------------------------------------------------------------------------
# Label taxonomy
# ----------------------------------------------------------------------------------------------------------------------


class TestSegmentationLabelTaxonomy:
    def test_wod_lidar_enum_size_and_default_mapping(self):
        assert len(WODPerceptionLidarSegmentationLabel) == 23
        # Every raw class maps to some unified default label.
        for label in WODPerceptionLidarSegmentationLabel:
            assert label.to_default() is not None

    def test_wod_camera_enum_size_and_default_mapping(self):
        assert len(WODPerceptionCameraSegmentationLabel) == 29  # 28 classes + TYPE_UNDEFINED
        for label in WODPerceptionCameraSegmentationLabel:
            assert label.to_default() is not None

    def test_pandaset_lidar_enum_size_and_default_mapping(self):
        # PandaSet semseg has 42 classes with raw ids 1..42 (no unlabeled/0 sentinel class).
        assert len(PandasetLidarSegmentationLabel) == 42
        assert min(int(label) for label in PandasetLidarSegmentationLabel) == 1
        assert max(int(label) for label in PandasetLidarSegmentationLabel) == 42
        for label in PandasetLidarSegmentationLabel:
            assert label.to_default() is not None

    def test_nuscenes_lidar_enum_size_and_default_mapping(self):
        # nuScenes-lidarseg / panoptic share a 32-class taxonomy with raw ids 0..31 (0 == noise).
        assert len(NuScenesLidarSegmentationLabel) == 32
        assert min(int(label) for label in NuScenesLidarSegmentationLabel) == 0
        assert max(int(label) for label in NuScenesLidarSegmentationLabel) == 31
        for label in NuScenesLidarSegmentationLabel:
            assert label.to_default() is not None


class TestLidarPathKwargsColumn:
    """The "path" store option must persist ``ParsedLidar.load_kwargs`` (e.g. the nuScenes panoptic
    path) so per-point segmentation can be re-read at API time, and tolerate its absence."""

    def _write_path_lidar(self, log_dir: Path, load_kwargs):
        metadata = LidarMergedMetadata({LidarID.LIDAR_TOP: LidarMetadata("top", LidarID.LIDAR_TOP)})
        writer = ArrowLidarWriter(
            log_dir=log_dir,
            metadata=metadata,
            log_metadata=make_log_metadata(),
            lidar_store_option="path",
            lidar_codec=None,
        )
        writer.write_modality(
            ParsedLidar(
                metadata=metadata,
                start_timestamp=Timestamp.from_us(1000),
                end_timestamp=Timestamp.from_us(51000),
                dataset_root="/data/nuscenes",
                relative_path="samples/LIDAR_TOP/x.pcd.bin",
                load_kwargs=load_kwargs,
            )
        )
        writer.close()
        table = pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()
        return table, metadata.modality_key

    def test_kwargs_roundtrip(self, tmp_path: Path):
        load_kwargs = {
            "lidarseg_relative_path": "lidarseg/v1.0-mini/abc_lidarseg.bin",
            "panoptic_relative_path": "panoptic/v1.0-mini/abc_panoptic.npz",
        }
        table, key = self._write_path_lidar(tmp_path, load_kwargs)
        assert f"{key}.kwargs" in table.schema.names
        assert _read_load_kwargs(table, 0, key) == load_kwargs

    def test_kwargs_null_when_absent(self, tmp_path: Path):
        # Datasets without per-frame extras (or non-keyframe sweeps) write a null cell → read as None.
        table, key = self._write_path_lidar(tmp_path, None)
        assert _read_load_kwargs(table, 0, key) is None
