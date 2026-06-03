"""Download utilities for the nuReasoning dataset (Hugging Face ``qixuewei/nuReasoning``).

The dataset is public and stores **each clip as an individual ``.zip``**::

    qixuewei/nuReasoning/
    ├── README.md, data_schema.py, ...
    └── data/
        └── <split>/                       (train | validation | test)
            └── <part>/                    (part_1, part_2, ...)
                └── <log_name>_<keyframe_token>.zip

Each zip unpacks into a per-clip directory::

    <log_name>_<keyframe_token>/
    ├── metadata.json
    ├── map.pkl
    ├── ego_state/<timestamp_us>.pkl
    ├── annotations/<timestamp_us>.pkl
    ├── reasoning/<timestamp_us>.json
    └── ... sensor assets (cameras/, lidar/) referenced by metadata.json

This module exposes :class:`NureasoningDownloader` (Hydra-instantiable) which powers
both entry points, modeled on :mod:`py123d.parser.ncore.ncore_download` (live tree
listing + per-clip selection) and :mod:`py123d.parser.pandaset.pandaset_download`
(zip extraction):

1. ``py123d-download dataset=nureasoning`` — :meth:`download` fetches every selected
   clip's zip into a session-scoped :class:`tempfile.TemporaryDirectory`, extracts it
   into ``output_dir/<split>/<part>/<clip>/``, and deletes the zip. The repo's leading
   ``data/`` prefix is stripped so ``output_dir`` *is* the local ``data/`` root
   (== ``nureasoning_data_root``).

2. The :class:`~py123d.parser.nureasoning.nureasoning_parser.NureasoningParser`
   streaming path — each log parser calls :meth:`download_single_clip` to drop its
   assigned clip into a per-clip :class:`tempfile.TemporaryDirectory`, converts it,
   and deletes the temp dir before moving on.

Selection is incremental: pick ``splits`` (train/validation/test), ``parts``
(part_1, ...), explicit ``log_names``, or the first/random ``num_logs``. The repo tree
is enumerated live via :class:`huggingface_hub.HfApi` so this stays correct as more
splits/parts are uploaded. Nothing is written to the HuggingFace hub cache — zips are
always fetched directly into a ``local_dir`` we control.
"""

from __future__ import annotations

import logging
import os
import random as _random_mod
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from tqdm import tqdm

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.nureasoning.utils.nureasoning_constants import (
    NUREASONING_REPO_DATA_DIR,
    NUREASONING_REPO_ID,
    NUREASONING_REPO_TYPE,
)

logger = logging.getLogger(__name__)

_ZIP_SUFFIX = ".zip"


def _require_hf_hub():
    """Lazy import — ``huggingface_hub`` is only needed once a download is requested."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for nuReasoning downloads. Install it with:\n"
            "  pip install py123d[hf]\n"
            "or directly:\n"
            "  pip install huggingface_hub\n"
        ) from exc
    return HfApi, hf_hub_download


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the HF token from (in order): explicit arg, ``$HF_TOKEN``, ``$HUGGINGFACE_HUB_TOKEN``.

    The ``qixuewei/nuReasoning`` dataset is public, so ``None`` is fine; a token is only
    needed if the user pins a private fork.
    """
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _is_repo_dir(entry: object) -> bool:
    """Return ``True`` if a ``list_repo_tree`` entry is a folder (``RepoFolder``).

    Detected by class name to avoid depending on a specific ``huggingface_hub`` export.
    """
    return type(entry).__name__ == "RepoFolder"


def extract_nureasoning_clip(zip_path: Path, clip_dir: Path) -> Path:
    """Extract a per-clip ``.zip`` into ``clip_dir`` and return it.

    The clip's files (``metadata.json``, ``ego_state/``, ``annotations/``, ...) are
    written directly under ``clip_dir`` so the result matches what
    :class:`~py123d.parser.nureasoning.nureasoning_parser.NureasoningParser` expects
    (``clip_dir/metadata.json``, ``clip_dir/ego_state/...``).

    The upstream zips wrap everything in a single ``<clip_name>/`` folder (verified
    against the repo), so that wrapper is stripped. The strip also tolerates a wrapper
    under a different name, and no-ops for a flat (root-level) archive.

    :param zip_path: Path to the downloaded clip zip.
    :param clip_dir: Destination directory for the clip's contents (its ``.name`` is the
        clip name, used to recognize the ``<clip_name>/`` wrapper).
    :return: ``clip_dir``.
    """
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if not infos:
            raise RuntimeError(f"Archive {zip_path} contains no files; it may be corrupt.")

        # The upstream archives nest everything under a single "<clip_name>/" folder; strip it so
        # files land directly under clip_dir. Prefer the known case (wrapper == clip name), and fall
        # back to a generic single-wrapper strip. A flat archive (multiple top-levels) is left as-is.
        top_levels = {info.filename.split("/", 1)[0] for info in infos}
        strip_prefix: Optional[str] = None
        if top_levels == {clip_dir.name}:
            strip_prefix = clip_dir.name + "/"
        elif len(top_levels) == 1 and all("/" in info.filename for info in infos):
            strip_prefix = next(iter(top_levels)) + "/"

        for info in infos:
            relative = info.filename
            if strip_prefix and relative.startswith(strip_prefix):
                relative = relative[len(strip_prefix) :]
            if not relative:
                continue
            dest = clip_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                # A plain copy loop keeps the dependency list clean (cf. extract_pandaset_log).
                while True:
                    chunk = src.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    dst.write(chunk)

    return clip_dir


@dataclass(frozen=True)
class NureasoningClipEntry:
    """Picklable locator for one clip zip in the repo.

    :ivar split: HF split directory (``train`` / ``validation`` / ``test``).
    :ivar part: Part directory (``part_1``, ...).
    :ivar clip_name: Clip directory name ``<log_name>_<keyframe_token>`` (no ``.zip``).
    :ivar repo_path: Full path inside the repo, ``data/<split>/<part>/<clip>.zip``.
    """

    split: str
    part: str
    clip_name: str
    repo_path: str

    @property
    def sort_key(self) -> tuple:
        return (self.split, self.part, self.clip_name)


def _repo_path_for(split: str, part: str, clip_name: str) -> str:
    """Build the in-repo zip path ``data/<split>/<part>/<clip>.zip``."""
    return f"{NUREASONING_REPO_DATA_DIR}/{split}/{part}/{clip_name}{_ZIP_SUFFIX}"


# ======================================================================================
# Downloader (Hydra-instantiable, shared by py123d-download and the streaming parser)
# ======================================================================================


class NureasoningDownloader(BaseDownloader):
    """Downloader for the nuReasoning dataset via Hugging Face ``qixuewei/nuReasoning``.

    Operates in two modes:

    * :meth:`download` — bulk-fetch every selected clip, extracting each into
      ``output_dir/<split>/<part>/<clip>/``. Used by ``py123d-download dataset=nureasoning``.
    * :meth:`download_single_clip` — fetch one clip into a caller-provided directory.
      Used by :class:`~py123d.parser.nureasoning.nureasoning_parser.NureasoningParser`
      in streaming mode to drop each clip into a per-clip temp directory.

    The instance is picklable (simple attrs only) so it can be embedded in log-parser
    objects shipped across a Ray process pool.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        revision: str = "main",
        hf_token: Optional[str] = None,
        splits: Optional[List[str]] = None,
        parts: Optional[List[str]] = None,
        log_names: Optional[List[str]] = None,
        num_logs: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        max_workers: int = 8,
        keep_zip: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the nuReasoning downloader.

        :param output_dir: Destination for :meth:`download` — the local ``data/`` root
            (== ``nureasoning_data_root``). Selected clips land at
            ``output_dir/<split>/<part>/<clip>/``. Ignored by :meth:`download_single_clip`
            (which takes its own ``output_dir`` arg).
        :param revision: HuggingFace dataset branch, tag, or commit.
        :param hf_token: HF access token. Resolves through :func:`resolve_hf_token` —
            falls back to ``$HF_TOKEN`` / ``$HUGGINGFACE_HUB_TOKEN`` when ``None``. The
            dataset is public, so a token is not required.
        :param splits: Restrict to these HF split directories (``train`` / ``validation``
            / ``test``). ``None`` (default) discovers and uses every split in the repo.
        :param parts: Restrict to these part directories (``part_1``, ...) within the
            selected splits. ``None`` (default) uses every part present.
        :param log_names: Explicit clip names ``<log_name>_<keyframe_token>``. Mutually
            exclusive with ``num_logs``. Validated against the (split/part-filtered) repo
            listing.
        :param num_logs: Select the first N clips (or N random clips when
            ``sample_random=True``) from the filtered catalog.
        :param sample_random: Randomize ``num_logs`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param max_workers: Parallel clip download/extract workers.
        :param keep_zip: When ``True``, also keep each downloaded ``.zip`` next to its
            extracted directory. Default ``False`` extracts then discards the zip
            (roughly halves peak disk use).
        :param dry_run: If ``True``, :meth:`download` logs the plan without fetching.
        """
        if log_names and num_logs is not None:
            raise ValueError("log_names and num_logs are mutually exclusive.")
        if num_logs is not None and num_logs <= 0:
            raise ValueError("num_logs must be a positive integer.")

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        # Public config — also read by the streaming parser.
        self.revision: str = revision
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.max_workers: int = max_workers
        self.keep_zip: bool = keep_zip

        # Selection knobs — consumed by :meth:`resolve_zip_entries`.
        self._splits: Optional[List[str]] = list(splits) if splits else None
        self._parts: Optional[List[str]] = list(parts) if parts else None
        self._explicit_log_names: Optional[List[str]] = list(log_names) if log_names else None
        self._num_logs: Optional[int] = num_logs
        self._sample_random: bool = sample_random
        self._seed: int = seed

    # ----- Selection ------------------------------------------------------------------

    def _list_child_dirs(self, api, path_in_repo: str) -> List[str]:
        """Return the immediate child folder names under ``path_in_repo`` (sorted)."""
        try:
            entries = api.list_repo_tree(
                repo_id=NUREASONING_REPO_ID,
                repo_type=NUREASONING_REPO_TYPE,
                path_in_repo=path_in_repo,
                revision=self.revision,
                recursive=False,
            )
            return sorted(Path(e.path).name for e in entries if _is_repo_dir(e))
        except Exception as exc:  # noqa: BLE001 — surface as a warning, keep enumerating other paths.
            logger.warning("nuReasoning: could not list %r (%s); skipping.", path_in_repo, exc)
            return []

    def _list_zip_entries(self) -> List[NureasoningClipEntry]:
        """Enumerate every selected clip zip in the repo via the HuggingFace tree API."""
        HfApi, _ = _require_hf_hub()
        api = HfApi(token=self.hf_token)
        data_dir = NUREASONING_REPO_DATA_DIR

        splits = self._splits if self._splits is not None else self._list_child_dirs(api, data_dir)
        if self._splits is None:
            logger.info("nuReasoning: discovered splits %s in %s", splits, NUREASONING_REPO_ID)

        entries: List[NureasoningClipEntry] = []
        for split in splits:
            split_path = f"{data_dir}/{split}"
            parts = self._parts if self._parts is not None else self._list_child_dirs(api, split_path)
            for part in parts:
                part_path = f"{split_path}/{part}"
                try:
                    children = list(
                        api.list_repo_tree(
                            repo_id=NUREASONING_REPO_ID,
                            repo_type=NUREASONING_REPO_TYPE,
                            path_in_repo=part_path,
                            revision=self.revision,
                            recursive=False,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("nuReasoning: could not list %r (%s); skipping.", part_path, exc)
                    continue

                for child in children:
                    if _is_repo_dir(child) or not child.path.endswith(_ZIP_SUFFIX):
                        continue
                    clip_name = Path(child.path).name[: -len(_ZIP_SUFFIX)]
                    entries.append(
                        NureasoningClipEntry(split=split, part=part, clip_name=clip_name, repo_path=child.path)
                    )
        return entries

    def resolve_zip_entries(self) -> List[NureasoningClipEntry]:
        """Return the clip entries selected by the current configuration (deterministic).

        Used by :meth:`download` for bulk extraction and by the streaming parser to
        enumerate work across Ray workers.
        """
        entries = sorted(self._list_zip_entries(), key=lambda e: e.sort_key)

        if self._explicit_log_names:
            wanted = set(self._explicit_log_names)
            found = {e.clip_name for e in entries}
            unknown = wanted - found
            if unknown:
                raise ValueError(
                    f"Unknown nuReasoning clip name(s): {sorted(unknown)}. Not found in the "
                    f"selected splits/parts of {NUREASONING_REPO_ID}@{self.revision}."
                )
            resolved = [e for e in entries if e.clip_name in wanted]
        elif self._num_logs is None or self._num_logs >= len(entries):
            resolved = entries
        elif self._sample_random:
            rng = _random_mod.Random(self._seed)
            resolved = sorted(rng.sample(entries, self._num_logs), key=lambda e: e.sort_key)
        else:
            resolved = entries[: self._num_logs]
        return resolved

    # ----- Bulk download (py123d-download) --------------------------------------------

    def download(self) -> None:
        """Inherited, see superclass.

        Bulk flow: every selected clip zip is downloaded into a session-scoped
        :class:`tempfile.TemporaryDirectory`, extracted into
        ``output_dir/<split>/<part>/<clip>/``, and (unless ``keep_zip``) the zip is
        discarded. Already-extracted clips are skipped, so re-running resumes.
        """
        entries = self.resolve_zip_entries()

        n_splits = len({e.split for e in entries})
        n_parts = len({(e.split, e.part) for e in entries})
        logger.info("nuReasoning source:    %s@%s", NUREASONING_REPO_ID, self.revision)
        logger.info(
            "nuReasoning selected:  %d clip(s) across %d split(s) / %d part(s)", len(entries), n_splits, n_parts
        )
        logger.info("nuReasoning target:    %s", self.output_dir)

        # dry_run previews the plan without writing, so it does not require output_dir.
        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d clip(s).", len(entries))
            return

        if not entries:
            logger.warning("No clips selected — nothing to download.")
            return

        assert self.output_dir is not None, "NureasoningDownloader.output_dir must be set before download()."
        self.output_dir.mkdir(parents=True, exist_ok=True)

        failures: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entry = {
                executor.submit(self._fetch_and_extract, entry, self._bulk_clip_dir(entry)): entry for entry in entries
            }
            for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="nuReasoning clips"):
                entry = future_to_entry[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 — collect and report all failures.
                    logger.error("Failed to download/extract clip %s: %s", entry.clip_name, exc)
                    failures.append(entry.clip_name)

        if failures:
            raise RuntimeError(
                f"{len(failures)} / {len(entries)} nuReasoning clip(s) failed to download: {failures[:10]}"
                + (" ..." if len(failures) > 10 else "")
            )
        logger.info("nuReasoning download complete: %s", self.output_dir)

    def _bulk_clip_dir(self, entry: NureasoningClipEntry) -> Path:
        """Destination clip directory for :meth:`download`: ``output_dir/<split>/<part>/<clip>``."""
        assert self.output_dir is not None
        return self.output_dir / entry.split / entry.part / entry.clip_name

    # ----- Per-clip fetch (streaming conversion) --------------------------------------

    def download_single_clip(self, split: str, part: str, clip_name: str, output_dir: Union[str, Path]) -> Path:
        """Fetch and extract one clip into ``output_dir/<clip_name>/`` and return that path.

        A convenience for materializing a single clip on demand (idempotent). The
        :class:`~py123d.parser.nureasoning.nureasoning_parser.NureasoningParser` streaming
        path uses :meth:`download` to materialize the whole selected subset at once
        (nuReasoning maps are per-log, so log and map parsers share one extracted tree),
        but this stays available for callers that want just one clip.
        """
        entry = NureasoningClipEntry(
            split=split, part=part, clip_name=clip_name, repo_path=_repo_path_for(split, part, clip_name)
        )
        clip_dir = Path(output_dir) / clip_name
        return self._fetch_and_extract(entry, clip_dir)

    def _fetch_and_extract(self, entry: NureasoningClipEntry, clip_dir: Path) -> Path:
        """Download ``entry``'s zip and extract it into ``clip_dir`` (idempotent)."""
        clip_dir = Path(clip_dir)
        if clip_dir.exists() and any(clip_dir.iterdir()):
            logger.debug("Skip already-extracted clip %s at %s", entry.clip_name, clip_dir)
            return clip_dir

        _, hf_hub_download = _require_hf_hub()
        with tempfile.TemporaryDirectory(prefix=f"py123d-nureasoning-{entry.clip_name}-") as tmp:
            local_zip = hf_hub_download(
                repo_id=NUREASONING_REPO_ID,
                repo_type=NUREASONING_REPO_TYPE,
                filename=entry.repo_path,
                revision=self.revision,
                token=self.hf_token,
                local_dir=tmp,
            )
            extract_nureasoning_clip(Path(local_zip), clip_dir)
            if self.keep_zip:
                clip_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_zip, clip_dir.parent / f"{entry.clip_name}{_ZIP_SUFFIX}")
        return clip_dir
