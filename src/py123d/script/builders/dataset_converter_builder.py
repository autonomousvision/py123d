import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_dataset_converters(cfg: DictConfig) -> List[AbstractDatasetConverter]:
    logger.info("Building AbstractDatasetConverter...")
    instantiated_datasets: List[AbstractDatasetConverter] = []
    for dataset_type in cfg.values():
        processor: AbstractDatasetConverter = instantiate(dataset_type)
        validate_type(processor, AbstractDatasetConverter)
        instantiated_datasets.append(processor)

    logger.info("Building AbstractDatasetConverter...DONE!")
    return instantiated_datasets
