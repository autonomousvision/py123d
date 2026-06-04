"""Unit tests for the nuScenes downloader catalog wiring and the lidarseg/panoptic extract path."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from py123d.parser.nuscenes.nuscenes_download import (
    _ARCHIVE_BY_NAME,
    NUSCENES_PRESETS,
    extract_nuscenes_archive,
)

_LIDARSEG_MINI = "nuScenes-lidarseg-mini-v1.0.tar.bz2"
_PANOPTIC_MINI = "nuScenes-panoptic-v1.0-mini.tar.gz"
_LIDARSEG_ALL = "nuScenes-lidarseg-all-v1.0.tar.bz2"
_PANOPTIC_ALL = "nuScenes-panoptic-v1.0-all.tar.gz"
_SEG_ARCHIVES = (_LIDARSEG_MINI, _PANOPTIC_MINI, _LIDARSEG_ALL, _PANOPTIC_ALL)


class TestSegmentationCatalog:
    def test_seg_archives_registered_as_autodetect_tar(self):
        for name in _SEG_ARCHIVES:
            assert name in _ARCHIVE_BY_NAME, f"{name} missing from the archive catalog"
            spec = _ARCHIVE_BY_NAME[name]
            # lidarseg is bz2, panoptic is gz — both go through the auto-detecting "tar" extractor.
            assert spec.extract_format == "tar"
            assert spec.md5 is None  # Motional publishes no checksums for the seg add-ons
            assert spec.category in ("lidarseg", "panoptic")

    def test_mini_preset_includes_mini_seg(self):
        mini = NUSCENES_PRESETS["mini"]
        assert _LIDARSEG_MINI in mini
        assert _PANOPTIC_MINI in mini
        # The smoketest must not pull the multi-GB all-splits archives.
        assert _LIDARSEG_ALL not in mini and _PANOPTIC_ALL not in mini

    def test_full_preset_includes_all_seg(self):
        full = NUSCENES_PRESETS["full"]
        for name in _SEG_ARCHIVES:
            assert name in full

    def test_every_preset_archive_is_in_catalog(self):
        for preset, archives in NUSCENES_PRESETS.items():
            for name in archives:
                assert name in _ARCHIVE_BY_NAME, f"preset {preset!r} references unknown archive {name!r}"


class TestTarExtraction:
    """The 'tar' format must transparently handle the lidarseg (.tar.bz2) and panoptic (.tar.gz) labels."""

    def _make_tar(self, path: Path, mode: str, member: str, payload: bytes) -> None:
        with tarfile.open(path, mode) as tar:
            info = tarfile.TarInfo(name=member)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    @pytest.mark.parametrize(
        "filename, mode",
        [
            ("nuScenes-lidarseg-mini-v1.0.tar.bz2", "w:bz2"),
            ("nuScenes-panoptic-v1.0-mini.tar.gz", "w:gz"),
            ("nuScenes-lidarseg-all-v1.0.tar", "w"),  # uncompressed tar must work too
        ],
    )
    def test_autodetect_extracts_seg_layout(self, tmp_path: Path, filename: str, mode: str):
        member = "lidarseg/v1.0-mini/token_lidarseg.bin"
        payload = b"\x01\x02\x03"
        archive = tmp_path / "archive" / filename
        archive.parent.mkdir(parents=True, exist_ok=True)
        self._make_tar(archive, mode, member, payload)

        out = tmp_path / "dataroot"
        extract_nuscenes_archive(archive, out, extract_format="tar")
        assert (out / member).read_bytes() == payload

    def test_unknown_format_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown extract_format"):
            extract_nuscenes_archive(tmp_path / "x.bin", tmp_path, extract_format="rar")
