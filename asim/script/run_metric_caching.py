import logging

import hydra
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig

from asim.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/metric_caching"
CONFIG_NAME = "default_metric_caching"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for metric caching.
    :param cfg: omegaconf dictionary
    """
    # Configure logger
    build_logger(cfg)

    # Build worker
    build_worker(cfg)

    # Precompute and cache all features
    logger.info("Starting Metric Caching...")
    # cache_data(cfg=cfg, worker=worker)


if __name__ == "__main__":
    main()
