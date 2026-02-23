import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.common.execution.executor import Executor
from py123d.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_executor(cfg: DictConfig) -> Executor:
    """
    Builds the executor.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of Executor.
    """
    logger.info("Building Executor...")
    executor: Executor = instantiate(cfg.execution)
    validate_type(executor, Executor)
    logger.info("Building Executor...DONE!")
    return executor
