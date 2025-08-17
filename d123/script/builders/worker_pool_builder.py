import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from d123.common.multithreading.worker_pool import WorkerPool
from d123.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info("Building WorkerPool...")
    worker: WorkerPool = instantiate(cfg.worker)
    validate_type(worker, WorkerPool)
    logger.info("Building WorkerPool...DONE!")
    return worker
