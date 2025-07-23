import logging
from typing import List

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from omegaconf import DictConfig

from d123.dataset.dataset_specific.raw_data_processor import RawDataProcessor

logger = logging.getLogger(__name__)


def build_data_processor(cfg: DictConfig) -> List[RawDataProcessor]:
    logger.info("Building RawDataProcessor...")
    instantiated_datasets: List[RawDataProcessor] = []
    for dataset_type in cfg.values():
        processor: RawDataProcessor = instantiate(dataset_type)
        validate_type(processor, RawDataProcessor)
        instantiated_datasets.append(processor)

    logger.info("Building RawDataProcessor...DONE!")
    return instantiated_datasets
