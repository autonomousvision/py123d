import logging
import pickle
from functools import partial
from pathlib import Path
from typing import List

import hydra
import lightning as L
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from d123.dataset.scene.abstract_scene import AbstractScene
from d123.script.builders.scene_builder_builder import build_scene_builder
from d123.script.builders.scene_filter_builder import build_scene_filter
from d123.script.run_dataset_conversion import build_worker
from d123.training.feature_builder.smart_feature_builder import SMARTFeatureBuilder

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
    logger.info(f"Found {len(scenes)} scenes.")

    cache_path = Path(cfg.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    feature_builder = SMARTFeatureBuilder()

    worker_map(worker, partial(_apply_feature_builder, feature_builder=feature_builder, cache_path=cache_path), scenes)


def _apply_feature_builder(scenes: List[AbstractScene], feature_builder: SMARTFeatureBuilder, cache_path: Path):

    for scene in scenes:
        scene.open()
        feature_dict = feature_builder.build_features(scene=scene)
        output_file = cache_path / f"{feature_dict['scenario_id']}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(feature_dict, f)
        scene.close()

    return []


if __name__ == "__main__":
    main()
