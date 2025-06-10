import logging
from typing import List

import hydra
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig

from asim.script.builders.data_processor_builder import RawDataProcessor, build_data_processor
from asim.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/dataset_caching"
CONFIG_NAME = "default_dataset_caching"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for metric caching.
    :param cfg: omegaconf dictionary
    """
    # Configure logger
    build_logger(cfg)

    # Build worker
    worker = build_worker(cfg)

    # Precompute and cache all features
    logger.info("Starting Dataset Caching...")
    data_processors: List[RawDataProcessor] = build_data_processor(cfg.datasets)
    for data_processor in data_processors:
        logger.info(f"Processing dataset: {data_processor.__class__.__name__}")
        data_processor.convert(worker=worker)
        logger.info(f"Finished processing dataset: {data_processor.__class__.__name__}")


if __name__ == "__main__":
    main()
