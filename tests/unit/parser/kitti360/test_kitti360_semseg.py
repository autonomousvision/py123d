"""Unit tests for KITTI-360 2D camera segmentation (semantic + panoptic/instance, sparse, path-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from py123d.datatypes import (
    CameraChannelType,
    CameraID,
    EgoStateSE3,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.modalities.base_modality import ModalityType
from py123d.geometry import PoseSE3
from py123d.parser.camera_segmentation_registry import Kitti360CameraSegmentationLabel
from py123d.parser.kitti360.kitti360_parser import (
    _extract_kitti360_camera_segmentation,
    _get_kitti360_segmentation_camera_metadata,
    _resolve_kitti360_semantic_relative_path,
)
from py123d.parser.kitti360.utils.kitti360_constants import (
    DIR_2D_SMT,
    DIR_2D_SMT_IMAGE_01,
    DIR_ROOT,
    KITTI360_EGO_STATE_SE3_METADATA,
    KITTI360_INSTANCE_SUBDIR,
    KITTI360_PINHOLE_CAMERA_IDS,
    KITTI360_SEMANTIC_SUBDIR,
)
from py123d.parser.kitti360.utils.kitti360_labels import id2label

LOG_NAME = "2013_05_28_drive_0000_sync"


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def _make_pinhole_metadata() -> Dict[CameraID, PinholeCameraMetadata]:
    return {
        camera_id: PinholeCameraMetadata(
            camera_name=camera_name,
            camera_id=camera_id,
            intrinsics=PinholeIntrinsics(fx=500.0, fy=500.0, cx=704.0, cy=188.0),
            distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
            width=1408,
            height=376,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        for camera_id, camera_name in KITTI360_PINHOLE_CAMERA_IDS.items()
    }


def _folders(root: Path) -> Dict[str, Path]:
    return {
        DIR_ROOT: root,
        DIR_2D_SMT: root / "data_2d_semantics",
        DIR_2D_SMT_IMAGE_01: root / "data_2d_semantics_image_01" / "data_2d_semantics",
    }


def _touch_label(folders: Dict[str, Path], root_key: str, camera_name: str, sub_dir: str, idx: int) -> Path:
    path = folders[root_key] / "train" / LOG_NAME / camera_name / sub_dir / f"{idx:010d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Taxonomy
# ----------------------------------------------------------------------------------------------------------------------


class TestKitti360CameraSegmentationTaxonomy:
    def test_enum_matches_kitti360_label_ids(self):
        # The enum must list exactly the KITTI-360 ``id`` values >= 0 (license plate, id -1, excluded).
        enum_ids = sorted(int(label) for label in Kitti360CameraSegmentationLabel)
        expected_ids = sorted(label_id for label_id in id2label if label_id >= 0)
        assert enum_ids == expected_ids
        assert len(enum_ids) == 45

    def test_every_class_maps_to_a_default(self):
        for label in Kitti360CameraSegmentationLabel:
            assert label.to_default() is not None


# ----------------------------------------------------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------------------------------------------------


class TestKitti360SegmentationMetadata:
    def test_semantic_metadata_for_both_pinhole_cameras(self):
        pinhole = _make_pinhole_metadata()
        meta = _get_kitti360_segmentation_camera_metadata(pinhole, CameraChannelType.SEMANTIC)
        assert set(meta) == set(KITTI360_PINHOLE_CAMERA_IDS)
        left = meta[CameraID.PCAM_STEREO_L]
        assert left.channel_type == CameraChannelType.SEMANTIC
        assert left.modality_type == ModalityType.CAMERA_SEMANTIC
        assert left.modality_key == "camera_semantic.pcam_stereo_l"
        assert left.segmentation_label_class is Kitti360CameraSegmentationLabel
        # Geometry delegated to the sibling RGB camera.
        assert (left.width, left.height) == (1408, 376)

    def test_instance_metadata_routes_to_distinct_file(self):
        pinhole = _make_pinhole_metadata()
        meta = _get_kitti360_segmentation_camera_metadata(pinhole, CameraChannelType.INSTANCE)
        right = meta[CameraID.PCAM_STEREO_R]
        assert right.channel_type == CameraChannelType.INSTANCE
        assert right.modality_type == ModalityType.CAMERA_INSTANCE
        assert right.modality_key == "camera_instance.pcam_stereo_r"


# ----------------------------------------------------------------------------------------------------------------------
# Path resolution (canonical + split image_01 layouts)
# ----------------------------------------------------------------------------------------------------------------------


class TestResolveSemanticPath:
    def test_resolves_canonical_layout(self, tmp_path: Path):
        folders = _folders(tmp_path)
        _touch_label(folders, DIR_2D_SMT, "image_00", KITTI360_SEMANTIC_SUBDIR, 250)
        rel = _resolve_kitti360_semantic_relative_path(folders, LOG_NAME, "image_00", KITTI360_SEMANTIC_SUBDIR, 250)
        assert rel == Path("data_2d_semantics/train") / LOG_NAME / "image_00" / "semantic" / "0000000250.png"

    def test_resolves_split_image_01_layout(self, tmp_path: Path):
        folders = _folders(tmp_path)
        _touch_label(folders, DIR_2D_SMT_IMAGE_01, "image_01", KITTI360_SEMANTIC_SUBDIR, 250)
        rel = _resolve_kitti360_semantic_relative_path(folders, LOG_NAME, "image_01", KITTI360_SEMANTIC_SUBDIR, 250)
        expected = (
            Path("data_2d_semantics_image_01/data_2d_semantics/train")
            / LOG_NAME
            / "image_01"
            / "semantic"
            / "0000000250.png"
        )
        assert rel == expected

    def test_returns_none_when_absent(self, tmp_path: Path):
        assert _resolve_kitti360_semantic_relative_path(_folders(tmp_path), LOG_NAME, "image_00", "semantic", 9) is None


# ----------------------------------------------------------------------------------------------------------------------
# Emission (sparse, per-frame)
# ----------------------------------------------------------------------------------------------------------------------


class TestExtractCameraSegmentation:
    def _ego_state(self) -> EgoStateSE3:
        return EgoStateSE3.from_imu(
            imu_se3=PoseSE3.identity(),
            metadata=KITTI360_EGO_STATE_SE3_METADATA,
            dynamic_state_se3=None,
            timestamp=Timestamp.from_us(0),
        )

    def _timestamps(self, n: int = 300) -> Dict[str, List[Timestamp]]:
        return {
            camera_name: [Timestamp.from_us(i) for i in range(n)]
            for camera_name in KITTI360_PINHOLE_CAMERA_IDS.values()
        }

    def _metadatas(self):
        pinhole = _make_pinhole_metadata()
        semantic = _get_kitti360_segmentation_camera_metadata(pinhole, CameraChannelType.SEMANTIC)
        instance = _get_kitti360_segmentation_camera_metadata(pinhole, CameraChannelType.INSTANCE)
        return semantic, instance

    def test_emits_semantic_and_instance_when_present(self, tmp_path: Path):
        folders = _folders(tmp_path)
        _touch_label(folders, DIR_2D_SMT, "image_00", KITTI360_SEMANTIC_SUBDIR, 250)
        _touch_label(folders, DIR_2D_SMT, "image_00", KITTI360_INSTANCE_SUBDIR, 250)
        semantic, instance = self._metadatas()

        parsed = _extract_kitti360_camera_segmentation(
            LOG_NAME, 250, self._timestamps(), folders, semantic, instance, self._ego_state()
        )
        # Only image_00 has files; image_01 has none → exactly the two image_00 streams.
        assert len(parsed) == 2
        keys = {p.metadata.modality_key for p in parsed}
        assert keys == {"camera_semantic.pcam_stereo_l", "camera_instance.pcam_stereo_l"}
        for p in parsed:
            assert p.timestamp.time_us == 250
            assert str(p.relative_path).startswith("data_2d_semantics/train/")

    def test_instance_optional_independently_of_semantic(self, tmp_path: Path):
        folders = _folders(tmp_path)
        _touch_label(folders, DIR_2D_SMT, "image_00", KITTI360_SEMANTIC_SUBDIR, 251)  # semantic only
        semantic, instance = self._metadatas()

        parsed = _extract_kitti360_camera_segmentation(
            LOG_NAME, 251, self._timestamps(), folders, semantic, instance, self._ego_state()
        )
        assert len(parsed) == 1
        assert parsed[0].metadata.channel_type == CameraChannelType.SEMANTIC

    def test_no_emission_for_unannotated_frame(self, tmp_path: Path):
        semantic, instance = self._metadatas()
        parsed = _extract_kitti360_camera_segmentation(
            LOG_NAME, 999, self._timestamps(), _folders(tmp_path), semantic, instance, self._ego_state()
        )
        assert parsed == []
