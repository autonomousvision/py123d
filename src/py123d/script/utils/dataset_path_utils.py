"""Backward-compatible shim for dataset path management.

Delegates to :mod:`py123d.common.dataset_paths`. Prefer importing from
``py123d.common.dataset_paths`` directly in new code.
"""

import logging

from omegaconf import DictConfig

from py123d.common.dataset_paths import DatasetPaths
from py123d.common.dataset_paths import get_dataset_paths as _get_dataset_paths
from py123d.common.dataset_paths import setup_dataset_paths as _setup_dataset_paths

logger = logging.getLogger(__name__)


def setup_dataset_paths(cfg: DictConfig) -> None:
    """Setup global dataset paths from a Hydra DictConfig.

    Converts the DictConfig to a :class:`DatasetPaths` dataclass, stores it globally,
    and exports to env vars for child process inheritance.

    :param cfg: The ``dataset_paths`` sub-config from Hydra.
    """
    paths = DatasetPaths.from_dict_config(cfg)
    _setup_dataset_paths(paths)
    paths.export_to_env()


def get_dataset_paths() -> DatasetPaths:
    """Get the global dataset paths.

    :return: :class:`DatasetPaths` instance.
    """
    return _get_dataset_paths()
