import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from d123.conversion.abstract_dataset_converter import AbstractLogWriter
from d123.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from d123.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_map_writer(cfg: DictConfig) -> AbstractMapWriter:
    logger.info("Building AbstractMapWriter...")
    map_writer: AbstractLogWriter = instantiate(cfg)
    validate_type(map_writer, AbstractMapWriter)
    logger.info("Building AbstractMapWriter...DONE!")
    return map_writer


def build_log_writer(cfg: DictConfig) -> AbstractLogWriter:
    logger.info("Building AbstractLogWriter...")
    log_writer: AbstractLogWriter = instantiate(cfg)
    validate_type(log_writer, AbstractLogWriter)
    logger.info("Building AbstractLogWriter...DONE!")
    return log_writer
