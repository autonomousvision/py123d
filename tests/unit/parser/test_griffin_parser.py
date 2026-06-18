"""Unit tests for the Griffin (vehicle-side) parser.

These build a tiny synthetic ``griffin-release`` tree on disk so the full parse
path - scene/split routing, calibration, pose, label, and LiDAR I/O, plus the
ego/global frame transforms - is exercised without the multi-GB real dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from py123d.datatypes import (
    BoxDetectionsSE3,
    CameraID,
    EgoStateSE3,
    LidarID,
)
from py123d.datatypes.modalities.base_modality import ModalityType
from py123d.parser.base_dataset_parser import ParsedCamera, ParsedLidar
from py123d.parser.griffin.griffin_parser import GriffinParser
from py123d.parser.griffin.griffin_sensor_io import load_griffin_point_cloud_data_from_path
from py123d.parser.griffin.utils.griffin_constants import (
    GRIFFIN_SPLITS,
    split_to_subset_and_kind,
)
from py123d.parser.griffin.utils.griffin_utils import parse_label_file
from py123d.parser.registry import GriffinBoxDetectionLabel

SUBSET = "griffin_50scenes_25m"
# Two scenes; only the first is in the synthetic "train" split file below.
SCENE_TRAIN = "scene-0000-TownTest-000"
SCENE_VAL = "scene-0001-TownTest-001"
FRAMES = ["000000", "000001", "000620"]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _identity_extrinsic() -> List[List[float]]:
    return np.eye(4).tolist()


def _write_ply(path: Path, points: np.ndarray, intensity: np.ndarray) -> None:
    """Write a minimal ASCII PLY with x, y, z and uppercase-I intensity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\nproperty float I\n"
        "end_header\n"
    )
    body = "".join(f"{x} {y} {z} {i}\n" for (x, y, z), i in zip(points, intensity))
    path.write_text(header + body)


@pytest.fixture()
def griffin_root(tmp_path: Path) -> Path:
    """Create a synthetic two-scene Griffin release with a custom split file."""
    data_root = tmp_path / "datasets"
    vehicle = data_root / SUBSET / "griffin-release" / "vehicle-side"

    # scene_infos.json: index 0 -> TownTest-000, index 1 -> TownTest-001.
    scene_infos = [
        {"name": "TownTest-000", "info": {"frames": FRAMES, "weather": "Clear"}},
        {"name": "TownTest-001", "info": {"frames": ["000010"], "weather": "Clear"}},
    ]
    _write_json(vehicle / "scene_infos.json", scene_infos)

    # Calibration: 4 cameras (identity extrinsic, simple intrinsics) + lidar_top.
    intrinsic = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    for cam in ["front", "back", "left", "right"]:
        _write_json(vehicle / "calib" / f"{cam}.json", {"extrinsic": _identity_extrinsic(), "intrinsic": intrinsic})
    _write_json(vehicle / "calib" / "lidar_top.json", {"extrinsic": _identity_extrinsic(), "intrinsic": None})

    # Per-frame pose, label, lidar, and camera images for the train scene's frames.
    for frame in FRAMES:
        _write_json(
            vehicle / "pose" / f"{frame}.json",
            {"x": 10.0, "y": 5.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 90.0},
        )
        # One car (traffic, kept) and one Soldier (non-traffic, skipped).
        label_lines = [
            "Car 3.0 0.0 0.0 4.5 1.9 1.5 0.0 0.0 0.0 7 0.9",
            "Soldier 1.0 1.0 0.0 0.5 0.5 1.8 0.0 0.0 0.0 8 1.0",
        ]
        (vehicle / "label").mkdir(parents=True, exist_ok=True)
        (vehicle / "label" / f"{frame}.txt").write_text("\n".join(label_lines) + "\n")

        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        inten = np.array([10.0, 20.0], dtype=np.float32)
        _write_ply(vehicle / "lidar" / "lidar_top" / f"{frame}.ply", pts, inten)

        for cam in ["front", "back", "left", "right"]:
            img = vehicle / "camera" / cam / f"{frame}.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"\x89PNG\r\n")  # placeholder; parser only references the path

    # Split file: only the train scene goes to train; the other to val.
    split_root = tmp_path / "split_datas"
    _write_json(split_root / f"{SUBSET}.json", {"batch_split": {"train": [SCENE_TRAIN], "val": [SCENE_VAL]}})

    return data_root


def _make_parser(griffin_root: Path, split: str) -> GriffinParser:
    return GriffinParser(
        splits=[split],
        griffin_data_root=griffin_root,
        split_data_root=griffin_root.parent / "split_datas",
    )


class TestSplitRouting:
    def test_all_splits_known(self):
        assert len(GRIFFIN_SPLITS) == 8
        assert "griffin_50scenes_25m_train" in GRIFFIN_SPLITS
        assert split_to_subset_and_kind("griffin_100scenes_random_val") == ("griffin_100scenes_random", "val")

    def test_unknown_split_raises(self):
        with pytest.raises(ValueError):
            split_to_subset_and_kind("griffin_50scenes_25m_test")

    def test_train_split_selects_only_train_scene(self, griffin_root: Path):
        parser = _make_parser(griffin_root, f"{SUBSET}_train")
        log_parsers = parser.get_log_parsers()
        assert len(log_parsers) == 1
        assert log_parsers[0].get_log_metadata().log_name == SCENE_TRAIN

    def test_val_split_selects_only_val_scene(self, griffin_root: Path):
        parser = _make_parser(griffin_root, f"{SUBSET}_val")
        log_parsers = parser.get_log_parsers()
        assert len(log_parsers) == 1
        assert log_parsers[0].get_log_metadata().log_name == SCENE_VAL

    def test_log_metadata_fields(self, griffin_root: Path):
        parser = _make_parser(griffin_root, f"{SUBSET}_train")
        meta = parser.get_log_parsers()[0].get_log_metadata()
        assert meta.dataset == "griffin"
        assert meta.split == f"{SUBSET}_train"
        assert meta.location == SUBSET

    def test_no_map(self, griffin_root: Path):
        assert _make_parser(griffin_root, f"{SUBSET}_train").get_map_parsers() == []


class TestModalitiesSync:
    def test_frame_count_and_timestamps(self, griffin_root: Path):
        log_parser = _make_parser(griffin_root, f"{SUBSET}_train").get_log_parsers()[0]
        frames = list(log_parser.iter_modalities_sync())
        assert len(frames) == len(FRAMES)
        # Frame "000620" -> 620 * 100_000 us = 62_000_000 us (62 s).
        assert frames[-1].timestamp.time_us == 620 * 100_000

    def test_modalities_present(self, griffin_root: Path):
        log_parser = _make_parser(griffin_root, f"{SUBSET}_train").get_log_parsers()[0]
        first = next(iter(log_parser.iter_modalities_sync()))
        kinds = [m.metadata.modality_type for m in first.modalities]
        assert ModalityType.EGO_STATE_SE3 in kinds
        assert ModalityType.LIDAR in kinds
        lidar = [m for m in first.modalities if isinstance(m, ParsedLidar)][0]
        cameras = [m for m in first.modalities if isinstance(m, ParsedCamera)]
        boxes = [m for m in first.modalities if isinstance(m, BoxDetectionsSE3)][0]
        assert lidar.metadata[LidarID.LIDAR_TOP].lidar_name == "lidar_top"
        assert len(cameras) == 4
        assert {c.metadata.camera_id for c in cameras} == {
            CameraID.PCAM_F0,
            CameraID.PCAM_B0,
            CameraID.PCAM_L0,
            CameraID.PCAM_R0,
        }
        # Only the traffic box survives (Soldier skipped); car -> VEHICLE.
        assert len(boxes.box_detections) == 1
        assert boxes.box_detections[0].attributes.label == GriffinBoxDetectionLabel.CAR

    def test_relative_paths_resolve_from_data_root(self, griffin_root: Path):
        log_parser = _make_parser(griffin_root, f"{SUBSET}_train").get_log_parsers()[0]
        first = next(iter(log_parser.iter_modalities_sync()))
        lidar = [m for m in first.modalities if isinstance(m, ParsedLidar)][0]
        rel = Path(lidar._relative_path)  # noqa: SLF001 (test introspection)
        assert rel.parts[0] == SUBSET
        assert (griffin_root / rel).exists()


class TestEgoPoseTransform:
    def test_yaw_90_rotation(self, griffin_root: Path):
        """yaw=90 deg should map ego +x to global +y (ENU, xyz Euler)."""
        log_parser = _make_parser(griffin_root, f"{SUBSET}_train").get_log_parsers()[0]
        ego: EgoStateSE3 = next(iter(log_parser.iter_modalities_sync())).modalities[0]
        imu = ego.imu_se3
        R = imu.rotation_matrix if hasattr(imu, "rotation_matrix") else imu.transformation_matrix[:3, :3]
        ego_x_in_global = R @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(ego_x_in_global, [0.0, 1.0, 0.0], atol=1e-6)
        assert np.allclose([imu.x, imu.y, imu.z], [10.0, 5.0, 0.0], atol=1e-9)


class TestLabelParsing:
    def test_12_11_9_field_variants(self, tmp_path: Path):
        f = tmp_path / "label.txt"
        f.write_text(
            "Car 1 2 3 4 5 6 7 8 9 10 0.5\n"  # 12 fields
            "Car 1 2 3 4 5 6 7 8 9 10\n"  # 11 fields (no visibility)
            "Car 1 2 3 4 5 6 9 10\n"  # 9 fields (no roll/pitch/visibility)
        )
        anns = parse_label_file(f)
        assert len(anns) == 3
        assert anns[0]["visibility"] == 0.5 and anns[0]["yaw"] == 9.0
        assert anns[1]["visibility"] == 1.0 and anns[1]["track_id"] == "10"
        assert anns[2]["roll"] == 0.0 and anns[2]["pitch"] == 0.0 and anns[2]["yaw"] == 9.0

    def test_missing_file_is_empty(self, tmp_path: Path):
        assert parse_label_file(tmp_path / "nope.txt") == []


class TestLidarIO:
    def test_uppercase_intensity_and_ego_frame(self, griffin_root: Path):
        ply = griffin_root / SUBSET / "griffin-release" / "vehicle-side" / "lidar" / "lidar_top" / f"{FRAMES[0]}.ply"
        points, features = load_griffin_point_cloud_data_from_path(ply)
        assert points.shape == (2, 3) and points.dtype == np.float32
        assert np.allclose(points[0], [1.0, 2.0, 3.0])  # returned unchanged (ego frame)
        from py123d.datatypes import LidarFeature

        assert np.allclose(features[LidarFeature.INTENSITY.serialize()], [10.0, 20.0])
        assert np.all(features[LidarFeature.IDS.serialize()] == int(LidarID.LIDAR_TOP))


class TestRobustness:
    def test_partial_cameras_are_skipped(self, griffin_root: Path):
        """If a camera image is missing for a frame, that camera is skipped, others remain."""
        vehicle = griffin_root / SUBSET / "griffin-release" / "vehicle-side"
        (vehicle / "camera" / "back" / f"{FRAMES[0]}.png").unlink()  # drop one camera image

        log_parser = _make_parser(griffin_root, f"{SUBSET}_train").get_log_parsers()[0]
        first = next(iter(log_parser.iter_modalities_sync()))
        cams = {c.metadata.camera_id for c in first.modalities if isinstance(c, ParsedCamera)}
        assert CameraID.PCAM_B0 not in cams  # back dropped
        assert len(cams) == 3  # front/left/right still present

    def test_empty_label_file_yields_no_boxes(self, griffin_root: Path):
        """A frame whose label file is empty still produces a valid (empty) detection set."""
        vehicle = griffin_root / SUBSET / "griffin-release" / "vehicle-side"
        (vehicle / "label" / f"{FRAMES[0]}.txt").write_text("")

        log_parser = _make_parser(griffin_root, f"{SUBSET}_train").get_log_parsers()[0]
        first = next(iter(log_parser.iter_modalities_sync()))
        boxes = [m for m in first.modalities if isinstance(m, BoxDetectionsSE3)][0]
        assert boxes.box_detections == []

    def test_log_ordering_is_deterministic(self, griffin_root: Path):
        """Repeated get_log_parsers calls return logs in the same order."""
        parser = _make_parser(griffin_root, f"{SUBSET}_train")
        names_a = [lp.get_log_metadata().log_name for lp in parser.get_log_parsers()]
        names_b = [lp.get_log_metadata().log_name for lp in parser.get_log_parsers()]
        assert names_a == names_b

    def test_malformed_extrinsic_rejected(self, tmp_path: Path):
        """A non-4x4 extrinsic is rejected rather than silently mis-parsed."""
        import json as _json

        from py123d.parser.griffin.utils.griffin_utils import load_calibration

        calib = tmp_path / "calib"
        calib.mkdir()
        (calib / "front.json").write_text(_json.dumps({"extrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))
        with pytest.raises(AssertionError):
            load_calibration(calib, "front")

    def test_missing_subset_dir_skips_gracefully(self, griffin_root: Path):
        """A split whose subset tree is absent yields no logs (warns, does not crash)."""
        parser = GriffinParser(
            splits=["griffin_100scenes_random_train"],  # not created in the fixture
            griffin_data_root=griffin_root,
            split_data_root=griffin_root.parent / "split_datas",
        )
        assert parser.get_log_parsers() == []
