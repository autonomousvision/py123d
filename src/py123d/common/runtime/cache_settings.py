"""API-layer cache settings management.

Provides a typed, frozen dataclass for cache-size knobs consumed by the read-time
API (e.g., the LRU mmap cache in :mod:`py123d.api.utils.arrow_helper`). Designed
to mirror :mod:`py123d.common.runtime.dataset_paths`:

- Environment variable resolution (works in any process, incl. Ray/forkserver
  children that inherit env vars from the main process).
- Hydra ``DictConfig`` conversion (main process entry point).
- Env var export so child processes pick up CLI/YAML overrides.
- A global singleton with lazy fallback to ``from_env()``, so library users
  (no Hydra) get sensible defaults without any setup.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

# Mapping from CacheSettings field name to environment variable name.
_ENV_VAR_MAP: Dict[str, str] = {
    "max_lru_cached_tables": "PY123D_MAX_LRU_CACHED_TABLES",
}


@dataclass(frozen=True)
class CacheSettings:
    """Immutable container for API-layer cache knobs.

    To add a new knob:
        1. Add an ``int`` field below with a sensible default.
        2. Add the env var mapping to ``_ENV_VAR_MAP``.
        3. Add an entry to ``default_cache_settings.yaml``.
        4. Read it via :func:`get_cache_settings` in the consuming module.
    """

    # Upper bound on memory-mapped Arrow tables held open by the global
    # _ArrowMmapStore in py123d.api.utils.arrow_helper.
    max_lru_cached_tables: int = 5_000

    @classmethod
    def from_env(cls) -> CacheSettings:
        """Create CacheSettings by reading environment variables.

        Works in any process because env vars are inherited by child processes.
        Missing or unparseable env vars fall back to dataclass defaults.
        """
        kwargs = {}
        for field_name, env_var in _ENV_VAR_MAP.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    kwargs[field_name] = int(value)
                except ValueError:
                    logger.warning(
                        "Ignoring env var %s=%r: cannot parse as int; using default.",
                        env_var,
                        value,
                    )
        return cls(**kwargs)

    @classmethod
    def from_dict_config(cls, cfg: object) -> CacheSettings:
        """Create CacheSettings from an OmegaConf DictConfig (Hydra integration).

        :param cfg: The ``cache_settings`` sub-config from Hydra.
        """
        kwargs = {}
        for f in fields(cls):
            value = getattr(cfg, f.name, None)
            if value is not None:
                kwargs[f.name] = int(value)
        return cls(**kwargs)

    def export_to_env(self) -> None:
        """Export field values to environment variables.

        Ensures child processes (forkserver, Ray) inherit Hydra CLI overrides.
        """
        for field_name, env_var in _ENV_VAR_MAP.items():
            value = getattr(self, field_name)
            os.environ[env_var] = str(value)


# ---------------------------------------------------------------------------
# Global accessor
# ---------------------------------------------------------------------------

_global_cache_settings: Optional[CacheSettings] = None


def setup_cache_settings(value: Union[CacheSettings, object]) -> None:
    """Set the global CacheSettings instance.

    Accepts either a :class:`CacheSettings` instance or any object with matching
    attributes (e.g., a Hydra ``DictConfig``), which is converted via
    :meth:`CacheSettings.from_dict_config`. Always exports values to env vars so
    forked / Ray child processes inherit them. Idempotent: only the first call
    takes effect.

    :param value: A :class:`CacheSettings` or a DictConfig-shaped object.
    """
    settings = value if isinstance(value, CacheSettings) else CacheSettings.from_dict_config(value)
    global _global_cache_settings  # noqa: PLW0603
    if _global_cache_settings is None:
        _global_cache_settings = settings
    settings.export_to_env()


def get_cache_settings() -> CacheSettings:
    """Get the global CacheSettings.

    If not explicitly set via :func:`setup_cache_settings`, creates one from
    environment variables. This works in child processes because env vars are
    inherited, and gives library users a usable default without any setup.

    :return: The global CacheSettings instance.
    """
    global _global_cache_settings  # noqa: PLW0603
    if _global_cache_settings is None:
        _global_cache_settings = CacheSettings.from_env()
    return _global_cache_settings
