import logging
import pickle
from functools import partial
from pathlib import Path
from typing import List

import hydra
import lightning as L

# from lightning.pytorch.loggers import Logger
# from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

from asim.dataset.dataset_specific.nuplan.nuplan_data_processor_ import worker_map
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.script.builders.scene_builder_builder import build_scene_builder
from asim.script.builders.scene_filter_builder import build_scene_filter
from asim.script.run_dataset_caching import build_worker
from asim.training.feature_builder.smart_feature_builder import SMARTFeatureBuilder

# from typing import List


# from lightning import Callback, LightningDataModule, LightningModule, Trainer


# from src.utils import (
#     RankedLogger,
#     instantiate_callbacks,
#     instantiate_loggers,
#     log_hyperparameters,
#     print_config_tree,
# )

# log = RankedLogger(__name__, rank_zero_only=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/preprocessing"
CONFIG_NAME = "default_preprocessing"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:

    L.seed_everything(cfg.seed, workers=True)

    worker = build_worker(cfg)
    scene_filter = build_scene_filter(cfg.scene_filter)
    scene_builder = build_scene_builder(cfg.scene_builder)

    scenes = scene_builder.get_scenes(scene_filter, worker=worker)
    logger.info(f"Found {len(scenes)} scenarios.")

    output_path = Path("/home/daniel/cache_test")
    output_path.mkdir(parents=True, exist_ok=True)

    feature_builder = SMARTFeatureBuilder()

    worker_map(worker, partial(_apply_feature_builder, feature_builder=feature_builder), scenes)


def _apply_feature_builder(
    scenes: List[AbstractScene],
    feature_builder: SMARTFeatureBuilder,
):

    output_path = Path("/home/daniel/cache_test")
    for scene in scenes:
        scene.open()
        feature_dict = feature_builder.build_features(scene=scene)
        output_file = output_path / f"{feature_dict['scenario_id']}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(feature_dict, f)
        scene.close()


if __name__ == "__main__":
    main()
