import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from d123.conversion.abstract_dataset_converter import AbstractLogWriter
from d123.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_log_writer(cfg: DictConfig) -> AbstractLogWriter:
    logger.info("Building AbstractLogWriter...")
    log_writer: AbstractLogWriter = instantiate(cfg)
    validate_type(log_writer, AbstractLogWriter)
    logger.info("Building AbstractLogWriter...DONE!")
    return log_writer
