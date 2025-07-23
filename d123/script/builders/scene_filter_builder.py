import logging
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from d123.dataset.scene.scene_filter import SceneFilter

logger = logging.getLogger(__name__)


def is_valid_token(token: Any) -> bool:
    """
    Basic check that a scene token is the right type/length.
    :token: parsed by hydra.
    :return: true if it looks valid, otherwise false.
    """
    if not isinstance(token, str) or len(token) != 16:
        return False

    try:
        return bytearray.fromhex(token).hex() == token
    except (TypeError, ValueError):
        return False


def build_scene_filter(cfg: DictConfig) -> SceneFilter:
    """
    Builds the scene filter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of SceneFilter.
    """
    logger.info("Building SceneFilter...")
    if cfg.scene_tokens and not all(map(is_valid_token, cfg.scene_tokens)):
        raise RuntimeError(
            "Expected all scene tokens to be 16-character strings. Your shell may strip quotes "
            "causing hydra to parse a token as a float, so consider passing them like "
            "scene_filter.scene_tokens='[\"595322e649225137\", ...]'"
        )
    scene_filter: SceneFilter = instantiate(cfg)
    assert isinstance(scene_filter, SceneFilter)
    logger.info("Building SceneFilter...DONE!")
    return scene_filter
