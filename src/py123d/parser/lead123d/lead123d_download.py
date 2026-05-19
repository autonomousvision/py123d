"""Download utilities for the ``lead123d`` dataset (pre-converted CARLA data on HuggingFace).

Unlike most py123d downloaders (which fetch raw upstream data to be converted later),
``lead123d`` already ships in py123d's native Arrow format — there is no separate
conversion step. The downloader fetches ``.arrow`` files directly into
``$PY123D_DATA_ROOT``, where the API expects them.

Dataset: https://huggingface.co/datasets/ln2697/lead123d

Top-level repo layout (only ``logs/`` and ``maps/`` are pulled — the ``data/``,
``results/``, ``scripts/``, ``stderr/``, ``stdout/`` folders and any JSON files are
skipped)::

    ln2697/lead123d/
    ├── logs/carla_train/{log_name}/*.arrow
    └── maps/carla/*.arrow

After download, files land at::

    $PY123D_DATA_ROOT/logs/carla_train/{log_name}/*.arrow
    $PY123D_DATA_ROOT/maps/carla/*.arrow

The downloader powers the ``py123d-download dataset=lead123d`` CLI; it does not have
a streaming-conversion mode because the data is already converted.
"""

from __future__ import annotations

import logging
import os
import random as _random_mod
from pathlib import Path
from typing import List, Optional, Tuple, Union

from py123d.parser.base_downloader import BaseDownloader

logger = logging.getLogger(__name__)

LEAD123D_REPO_ID = "ln2697/lead123d"
LEAD123D_REPO_TYPE = "dataset"
LEAD123D_LOGS_PREFIX = "logs"
LEAD123D_MAPS_PREFIX = "maps"

# Splits / map locations currently published on the HF mirror. Hard-coded because the
# repo layout is fixed; extend these tuples (and the downloader fields if needed) when
# new ones are added upstream.
DEFAULT_LOG_SPLITS: Tuple[str, ...] = ("carla_train",)
DEFAULT_MAP_LOCATIONS: Tuple[str, ...] = ("carla",)


def _require_hf_hub():
    """Lazy import — ``huggingface_hub`` is only needed once a download is requested."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for this module. Install it with:\n  pip install py123d[hf]\n"
        ) from exc
    return HfApi, snapshot_download


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the HF token from (in order): explicit arg, ``$HF_TOKEN``, ``$HUGGINGFACE_HUB_TOKEN``.

    The ``ln2697/lead123d`` mirror is public, so ``None`` is fine; a token is only
    needed for private forks or when hitting the anonymous rate limit.
    """
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def list_log_names(
    split: str,
    token: Optional[str] = None,
    revision: str = "main",
) -> List[str]:
    """List the log directory names directly under ``logs/{split}/`` in the repo.

    :param split: Logs split name (e.g. ``"carla_train"``).
    :param token: HuggingFace access token (optional for the public mirror).
    :param revision: Dataset branch/tag/commit.
    :return: Sorted list of log directory names (no ``logs/{split}/`` prefix).
    """
    HfApi, _ = _require_hf_hub()
    api = HfApi(token=token)
    entries = api.list_repo_tree(
        repo_id=LEAD123D_REPO_ID,
        repo_type=LEAD123D_REPO_TYPE,
        path_in_repo=f"{LEAD123D_LOGS_PREFIX}/{split}",
        revision=revision,
        recursive=False,
    )
    return sorted(Path(e.path).name for e in entries if e.path.startswith(f"{LEAD123D_LOGS_PREFIX}/{split}/"))


# ======================================================================================
# Downloader (Hydra-instantiable, used by py123d-download dataset=lead123d)
# ======================================================================================


class Lead123dDownloader(BaseDownloader):
    """Downloader for the pre-converted ``lead123d`` (CARLA) dataset on HuggingFace.

    Pulls ``.arrow`` files under ``logs/{split}/`` (one or more splits) and
    ``maps/{location}/`` directly into :attr:`output_dir` (typically
    ``$PY123D_DATA_ROOT``). Non-arrow files (JSON manifests etc.) and the unrelated
    top-level repo folders (``data/``, ``results/``, ``scripts/``, ``stderr/``,
    ``stdout/``) are skipped via ``allow_patterns``.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        revision: str = "main",
        hf_token: Optional[str] = None,
        log_names: Optional[List[str]] = None,
        num_logs: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        include_maps: bool = True,
        max_workers: int = 8,
        dry_run: bool = False,
    ) -> None:
        """Initialize the lead123d downloader.

        :param output_dir: Destination directory. Files are placed in the layout
            py123d expects: ``output_dir/logs/{split}/{log}/*.arrow``,
            ``output_dir/maps/{location}/*.arrow``. Defaults to
            ``$PY123D_DATA_ROOT`` via the Hydra config.
        :param revision: HuggingFace dataset branch, tag, or commit.
        :param hf_token: HF access token. Resolves through :func:`resolve_hf_token`
            — falls back to ``$HF_TOKEN`` / ``$HUGGINGFACE_HUB_TOKEN`` when ``None``.
            The mirror is public, so a token is not required.
        :param log_names: Explicit log directory names (across all default splits)
            to fetch. Mutually exclusive with ``num_logs``. When both are ``None``,
            the entire ``logs/{split}/`` tree is pulled with a single wildcard
            pattern (no catalog listing required).
        :param num_logs: First N (or N random when ``sample_random=True``) logs from
            the catalog of all configured splits. Triggers a catalog listing.
        :param sample_random: Randomize ``num_logs`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param include_maps: Whether to also pull ``maps/{location}/*.arrow``.
            Useful to set to ``False`` for log-only re-runs once maps are cached.
        :param max_workers: Parallel HF download workers.
        :param dry_run: If ``True``, :meth:`download` logs the plan without fetching.
        """
        if log_names and num_logs is not None:
            raise ValueError("log_names and num_logs are mutually exclusive.")
        if num_logs is not None and num_logs <= 0:
            raise ValueError("num_logs must be a positive integer.")

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        self.revision: str = revision
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.include_maps: bool = include_maps
        self.max_workers: int = max_workers

        self._explicit_log_names: Optional[List[str]] = list(log_names) if log_names else None
        self._num_logs: Optional[int] = num_logs
        self._sample_random: bool = sample_random
        self._seed: int = seed

    # ----- Selection ------------------------------------------------------------------

    def resolve_log_names(self) -> Optional[List[str]]:
        """Return the log directory names to fetch, or ``None`` for "every log".

        Returning ``None`` is the no-subset case: the caller short-circuits with a
        single wildcard ``logs/{split}/**/*.arrow`` pattern per split, avoiding the
        ``HfApi.list_repo_tree`` round-trip.
        """
        resolved: Optional[List[str]]
        if self._explicit_log_names:
            resolved = list(self._explicit_log_names)
        elif self._num_logs is None:
            resolved = None
        else:
            all_names: List[str] = []
            for split in DEFAULT_LOG_SPLITS:
                all_names.extend(list_log_names(split=split, token=self.hf_token, revision=self.revision))
            all_names = sorted(set(all_names))
            if self._num_logs >= len(all_names):
                resolved = all_names
            elif self._sample_random:
                rng = _random_mod.Random(self._seed)
                resolved = sorted(rng.sample(all_names, self._num_logs))
            else:
                resolved = all_names[: self._num_logs]
        return resolved

    def _build_allow_patterns(self, log_names: Optional[List[str]]) -> List[str]:
        """Build ``allow_patterns`` for ``snapshot_download``.

        :param log_names: Result of :meth:`resolve_log_names`. ``None`` means
            "every log under each split"; a list narrows to the named subset.
        """
        patterns: List[str] = []
        for split in DEFAULT_LOG_SPLITS:
            if log_names is None:
                patterns.append(f"{LEAD123D_LOGS_PREFIX}/{split}/**/*.arrow")
            else:
                for name in log_names:
                    patterns.append(f"{LEAD123D_LOGS_PREFIX}/{split}/{name}/*.arrow")
        if self.include_maps:
            for location in DEFAULT_MAP_LOCATIONS:
                patterns.append(f"{LEAD123D_MAPS_PREFIX}/{location}/*.arrow")
        return patterns

    # ----- Download -------------------------------------------------------------------

    def download(self) -> None:
        """Inherited, see superclass."""
        assert self.output_dir is not None, "Lead123dDownloader.output_dir must be set before download()."
        log_names = self.resolve_log_names()
        allow_patterns = self._build_allow_patterns(log_names=log_names)

        logger.info("lead123d source:        %s@%s", LEAD123D_REPO_ID, self.revision)
        logger.info("lead123d target dir:    %s", self.output_dir)
        logger.info(
            "lead123d log selection: %s",
            "all logs in splits=" + str(list(DEFAULT_LOG_SPLITS)) if log_names is None else f"{len(log_names)} log(s)",
        )
        logger.info("lead123d include_maps:  %s", self.include_maps)
        for pat in allow_patterns[: min(10, len(allow_patterns))]:
            logger.debug("  allow_pattern: %s", pat)

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d allow_pattern(s).", len(allow_patterns))
            return

        _, snapshot_download = _require_hf_hub()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=LEAD123D_REPO_ID,
            repo_type=LEAD123D_REPO_TYPE,
            revision=self.revision,
            local_dir=str(self.output_dir),
            allow_patterns=allow_patterns,
            token=self.hf_token,
            max_workers=self.max_workers,
        )
        logger.info("lead123d download complete: %s", self.output_dir)
