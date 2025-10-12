import gc
import logging
from functools import partial
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from d123 import ascii_banner
from d123.common.multithreading.worker_utils import worker_map
from d123.script.builders.dataset_converter_builder import AbstractDatasetConverter, build_dataset_converters
from d123.script.builders.worker_pool_builder import build_worker
from d123.script.builders.writer_builder import build_log_writer, build_map_writer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/conversion"
CONFIG_NAME = "default_conversion"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for metric caching.
    :param cfg: omegaconf dictionary
    """
    logger.info(ascii_banner)

    # Build worker

    # Precompute and cache all features
    logger.info("Starting Dataset Caching...")
    dataset_converters: List[AbstractDatasetConverter] = build_dataset_converters(cfg.datasets)

    for dataset_converter in dataset_converters:

        worker = build_worker(cfg)
        logger.info(f"Processing dataset: {dataset_converter.__class__.__name__}")

        map_args = [{"map_index": i} for i in range(dataset_converter.get_number_of_maps())]
        worker_map(worker, partial(_convert_maps, cfg=cfg, dataset_converter=dataset_converter), map_args)
        logger.info(f"Finished maps: {dataset_converter.__class__.__name__}")

        log_args = [{"log_index": i} for i in range(dataset_converter.get_number_of_logs())]
        worker_map(worker, partial(_convert_logs, cfg=cfg, dataset_converter=dataset_converter), log_args)
        logger.info(f"Finished logs: {dataset_converter.__class__.__name__}")

        logger.info(f"Finished processing dataset: {dataset_converter.__class__.__name__}")


def _convert_maps(args: List[Dict[str, int]], cfg: DictConfig, dataset_converter: AbstractDatasetConverter) -> List:

    map_writer = build_map_writer(cfg.map_writer)
    for arg in args:
        dataset_converter.convert_map(arg["map_index"], map_writer)
    return []


def _convert_logs(args: List[Dict[str, int]], cfg: DictConfig, dataset_converter: AbstractDatasetConverter) -> None:

    def _internal_convert_log(args: Dict[str, int], dataset_converter_: AbstractDatasetConverter) -> int:
        # for i2 in tqdm(range(300), leave=False)
        log_writer = build_log_writer(cfg.log_writer)
        for arg in args:
            dataset_converter_.convert_log(arg["log_index"], log_writer)
        del log_writer
        gc.collect()

    # for arg in :
    _internal_convert_log(args, dataset_converter)
    gc.collect()
    return []


if __name__ == "__main__":
    main()
