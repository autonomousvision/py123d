import logging
from typing import List

import hydra
from omegaconf import DictConfig

from d123.script.builders.data_converter_builder import RawDataConverter, build_data_converter
from d123.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/dataset_conversion"
CONFIG_NAME = "default_dataset_conversion"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for metric caching.
    :param cfg: omegaconf dictionary
    """

    # Build worker
    worker = build_worker(cfg)

    # Precompute and cache all features
    logger.info("Starting Dataset Caching...")
    data_processors: List[RawDataConverter] = build_data_converter(cfg.datasets)
    for data_processor in data_processors:

        logger.info(f"Processing dataset: {data_processor.__class__.__name__}")

        data_processor.convert_maps(worker=worker)
        logger.info(f"Finished maps: {data_processor.__class__.__name__}")

        data_processor.convert_logs(worker=worker)
        logger.info(f"Finished logs: {data_processor.__class__.__name__}")

        logger.info(f"Finished processing dataset: {data_processor.__class__.__name__}")


if __name__ == "__main__":
    main()
