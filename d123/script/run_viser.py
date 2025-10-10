import logging

import hydra
from omegaconf import DictConfig

from d123.common.visualization.viser.viser_viewer import ViserViewer
from d123.script.builders.scene_builder_builder import build_scene_builder
from d123.script.builders.scene_filter_builder import build_scene_filter
from d123.script.run_dataset_conversion import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/viser"
CONFIG_NAME = "default_viser"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:

    worker = build_worker(cfg)
    scene_filter = build_scene_filter(cfg.scene_filter)
    scene_builder = build_scene_builder(cfg.scene_builder)
    scenes = scene_builder.get_scenes(scene_filter, worker=worker)
<<<<<<< HEAD
    
    ViserVisualizationServer(scenes=scenes)
=======

    ViserViewer(scenes=scenes)
>>>>>>> dev_v0.0.7


if __name__ == "__main__":
    main()