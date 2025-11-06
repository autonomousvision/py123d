import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.script.builders.utils.utils_type import validate_type
from py123d.visualization.viser.viser_config import ViserConfig

logger = logging.getLogger(__name__)


def build_viser_config(cfg: DictConfig) -> ViserConfig:
    """
    Builds the config dataclass for the viser viewer.
    :param cfg: DictConfig. Configuration that is used to run the viewer.
    :return: Instance of ViserConfig.
    """
    logger.info("Building ViserConfig...")
    viser_config: ViserConfig = instantiate(cfg)
    validate_type(viser_config, ViserConfig)
    logger.info("Building ViserConfig...DONE!")
    return viser_config
