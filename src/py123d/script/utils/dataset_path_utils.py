import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)
_global_dataset_paths: Optional[DictConfig] = None


def setup_dataset_paths(cfg: DictConfig) -> None:
    """Setup the global dataset paths.

    :param cfg: The configuration containing dataset paths.
    :return: None
    """

    global _global_dataset_paths

    if _global_dataset_paths is None:
        # Make it immutable
        OmegaConf.set_struct(cfg, True)  # Prevents adding new keys
        OmegaConf.set_readonly(cfg, True)  # Prevents any modifications
        _global_dataset_paths = cfg

    return None


def get_dataset_paths() -> DictConfig:
    """Get the global dataset paths from anywhere in your code.

    :return: global dataset paths configuration
    """
    global _global_dataset_paths

    if _global_dataset_paths is None:
        dataset_paths_config_yaml = Path(__file__).parent.parent / "config" / "common" / "default_dataset_paths.yaml"
        logger.warning(f"Dataset paths not set. Using default config: {dataset_paths_config_yaml}")

        cfg = OmegaConf.load(dataset_paths_config_yaml)
        setup_dataset_paths(cfg.dataset_paths)

    return _global_dataset_paths
