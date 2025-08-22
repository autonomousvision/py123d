import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from d123.dataset.dataset_specific.raw_data_converter import RawDataConverter
from d123.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_data_converter(cfg: DictConfig) -> List[RawDataConverter]:
    logger.info("Building RawDataProcessor...")
    instantiated_datasets: List[RawDataConverter] = []
    for dataset_type in cfg.values():
        processor: RawDataConverter = instantiate(dataset_type)
        validate_type(processor, RawDataConverter)
        instantiated_datasets.append(processor)

    logger.info("Building RawDataProcessor...DONE!")
    return instantiated_datasets
