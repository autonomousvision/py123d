import logging
from typing import List

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/lightning_training"
CONFIG_NAME = "default_lightning_training"


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.callbacks)

    # logger.info(f"Instantiating loggers...")
    # logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    # # setup model watching
    # for _logger in logger:
    #     if isinstance(_logger, WandbLogger):
    #         _logger.watch(model, log="all")

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    logger.info(f"Resuming from ckpt: cfg.ckpt_path={cfg.ckpt_path}")
    if cfg.action == "fit":
        logger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    # elif cfg.action == "finetune":
    #     logger.info("Starting finetuning!")
    #     model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"], strict=False)
    #     trainer.fit(model=model, datamodule=datamodule)
    # elif cfg.action == "validate":
    #     logger.info("Starting validating!")
    #     trainer.validate(
    #         model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    #     )
    # elif cfg.action == "test":
    #     logger.info("Starting testing!")
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
