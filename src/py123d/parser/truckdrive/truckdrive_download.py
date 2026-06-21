"""Download utilities for the public TruckDrive dataset."""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.truckdrive.truckdrive_constants import (
    CLOUDFRONT_BASE_URL,
    DEFAULT_MODALITIES,
    MODALITY_ZIP_FILES,
    S3_PREFIX,
)

logger = logging.getLogger(__name__)

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class TruckDriveDownloader(BaseDownloader):
    """Downloader for the public TruckDrive CloudFront release."""

    def __init__(
        self,
        output_dir: Optional[Path],
        dry_run: bool = False,
        scenes: Optional[Sequence[str]] = None,
        modalities: Optional[Sequence[str]] = None,
        max_workers: int = 4,
        unzip: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Initialize the TruckDrive downloader.

        :param output_dir: Destination directory for downloaded scenes.
        :param dry_run: When ``True``, log the plan without downloading.
        :param scenes: Explicit scene names to download. When ``None``, all scenes are listed.
        :param modalities: Modalities to download. Defaults to camera/lidar/poses/calibrations/annotations.
        :param max_workers: Parallel download workers.
        :param unzip: Extract modality zips into the viewer layout after download.
        :param overwrite: Re-download files even when they already exist locally.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.dry_run = dry_run
        self.scenes = list(scenes) if scenes is not None else None
        self.modalities = tuple(modalities) if modalities is not None else DEFAULT_MODALITIES
        self.max_workers = max_workers
        self.unzip = unzip
        self.overwrite = overwrite

    def download(self) -> None:
        """Fetch selected TruckDrive scenes into :attr:`output_dir`."""
        if self.output_dir is None:
            raise ValueError("TruckDriveDownloader.output_dir must be set before calling download().")

        selected_modalities = self._resolve_modalities()
        scene_names = self.scenes if self.scenes else self._list_scene_names()
        if not scene_names:
            raise RuntimeError("No TruckDrive scenes found to download.")

        selected_keys: List[str] = []
        for scene_name in scene_names:
            scene_prefix = f"{S3_PREFIX}{scene_name}/"
            for key in self._list_keys_for_prefix(scene_prefix):
                filename = Path(key).name
                if filename in selected_modalities:
                    selected_keys.append(key)

        selected_keys = sorted(set(selected_keys))
        if not selected_keys:
            raise RuntimeError("No TruckDrive files matched the selected scenes/modalities.")

        logger.info("TruckDrive download plan: %d files under %s", len(selected_keys), self.output_dir)
        for key in selected_keys:
            logger.info("  %s", key)

        if self.dry_run:
            logger.info("Dry run enabled; skipping download.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_key, key) for key in selected_keys]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        if self.unzip:
            for key in selected_keys:
                self._unzip_modality_archive(self.output_dir / key)

    def _resolve_modalities(self) -> Tuple[str, ...]:
        unknown = [modality for modality in self.modalities if modality not in MODALITY_ZIP_FILES]
        if unknown:
            raise ValueError(f"Unknown TruckDrive modalities: {unknown}")
        return tuple(MODALITY_ZIP_FILES[modality] for modality in self.modalities)

    def _list_scene_names(self) -> List[str]:
        prefixes = self._list_prefixes_for_prefix(S3_PREFIX, delimiter="/")
        scene_names: List[str] = []
        for prefix in prefixes:
            remainder = prefix[len(S3_PREFIX) :].strip("/")
            if remainder.startswith("scene_"):
                scene_names.append(remainder)
        return sorted(set(scene_names))

    def _list_prefixes_for_prefix(self, prefix: str, delimiter: str = "/") -> List[str]:
        encoded_prefix = urllib.parse.quote(prefix, safe="")
        encoded_delimiter = urllib.parse.quote(delimiter, safe="")
        prefixes: List[str] = []
        marker: Optional[str] = None

        while True:
            url = f"{CLOUDFRONT_BASE_URL}/?prefix={encoded_prefix}&delimiter={encoded_delimiter}"
            if marker:
                url += f"&marker={urllib.parse.quote(marker, safe='')}"
            xml_text = self._fetch_text(url)
            root = ET.fromstring(xml_text)
            for elem in root.findall(".//s3:CommonPrefixes", _S3_NS):
                prefix_elem = elem.find("s3:Prefix", _S3_NS)
                if prefix_elem is not None and prefix_elem.text:
                    prefixes.append(prefix_elem.text)
            marker_elem = root.find(".//s3:NextMarker", _S3_NS)
            marker = marker_elem.text if marker_elem is not None else None
            if not marker:
                break
        return prefixes

    def _list_keys_for_prefix(self, prefix: str) -> List[str]:
        encoded_prefix = urllib.parse.quote(prefix, safe="")
        keys: List[str] = []
        marker: Optional[str] = None

        while True:
            url = f"{CLOUDFRONT_BASE_URL}/?prefix={encoded_prefix}&delimiter="
            if marker:
                url += f"&marker={urllib.parse.quote(marker, safe='')}"
            xml_text = self._fetch_text(url)
            root = ET.fromstring(xml_text)
            for elem in root.findall(".//s3:Key", _S3_NS):
                if elem.text:
                    keys.append(elem.text)
            marker_elem = root.find(".//s3:NextMarker", _S3_NS)
            marker = marker_elem.text if marker_elem is not None else None
            if not marker:
                break
        return keys

    def _fetch_text(self, url: str) -> str:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read().decode("utf-8")

    def _download_key(self, key: str) -> None:
        assert self.output_dir is not None
        destination = self.output_dir / key
        if destination.exists() and not self.overwrite:
            logger.info("[skip] %s", key)
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded_key = urllib.parse.quote(key, safe="/")
        url = f"{CLOUDFRONT_BASE_URL}/{encoded_key}"
        part_path = destination.with_suffix(destination.suffix + ".part")
        logger.info("[download] %s", key)
        with urllib.request.urlopen(url, timeout=120) as response, part_path.open("wb") as part_file:
            shutil.copyfileobj(response, part_file)
        part_path.replace(destination)

    def _unzip_modality_archive(self, zip_path: Path) -> None:
        if not zip_path.is_file():
            logger.warning("[unzip] skip missing archive: %s", zip_path)
            return

        scene_dir = zip_path.parent
        modality = zip_path.stem
        scene_name = scene_dir.name
        destination = scene_dir / modality
        if destination.is_dir() and any(destination.iterdir()):
            logger.info("[unzip] skip existing: %s", destination)
            return

        logger.info("[unzip] %s", zip_path)
        with tempfile.TemporaryDirectory(prefix="truckdrive_unzip_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(zip_path) as archive:
                if not archive.namelist():
                    return
                archive.extractall(tmp_path)

            matches = [path for path in tmp_path.rglob(modality) if path.is_dir() and path.name == modality]
            scoped = [path for path in matches if scene_name in path.parts]
            if scoped:
                matches = scoped
            if len(matches) == 1:
                self._move_tree(matches[0], destination)
                return

            direct = tmp_path / modality
            if direct.is_dir():
                self._move_tree(direct, destination)
                return

            if any(tmp_path.iterdir()):
                self._merge_tree(tmp_path, destination)
                return

        raise RuntimeError(f"Could not determine extracted layout for archive: {zip_path}")

    @staticmethod
    def _move_tree(source: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))

    @staticmethod
    def _merge_tree(source_dir: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source_dir.iterdir()):
            target = destination / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
