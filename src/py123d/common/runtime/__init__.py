"""Process-wide runtime configuration.

Bundles dataset paths and cache settings under one package. Hydra entry
points should call :func:`setup_runtime` once at startup; library users can
import the individual :class:`CacheSettings` / :class:`DatasetPaths`
dataclasses and their ``setup_*`` / ``get_*`` accessors directly.
"""

from py123d.common.runtime.cache_settings import (
    CacheSettings,
    get_cache_settings,
    setup_cache_settings,
)
from py123d.common.runtime.dataset_paths import (
    DatasetPaths,
    get_dataset_paths,
    setup_dataset_paths,
)

__all__ = [
    "CacheSettings",
    "DatasetPaths",
    "get_cache_settings",
    "get_dataset_paths",
    "setup_cache_settings",
    "setup_dataset_paths",
    "setup_runtime",
]


def setup_runtime(cfg: object) -> None:
    """Configure all global runtime state from a Hydra DictConfig.

    Reads ``cfg.dataset_paths`` and ``cfg.cache_settings`` and applies them to
    the respective global singletons. Each underlying ``setup_*`` is idempotent.

    :param cfg: The top-level Hydra DictConfig.
    """
    setup_dataset_paths(cfg.dataset_paths)
    setup_cache_settings(cfg.cache_settings)
