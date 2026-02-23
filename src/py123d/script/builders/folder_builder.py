import logging
import pathlib

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_experiment_folder(cfg: DictConfig) -> None:
    """
    Builds the main experiment folder.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    logger.info("Building experiment folders...")
    main_exp_folder = pathlib.Path(cfg.output_dir)
    logger.info(f"Experimental folder: {main_exp_folder}")
    main_exp_folder.mkdir(parents=True, exist_ok=True)
