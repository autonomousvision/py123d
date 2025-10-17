import logging

import hydra
from omegaconf import DictConfig

from py123d.common.visualization.viser.viser_viewer import ViserViewer
from py123d.script.builders.scene_builder_builder import build_scene_builder
from py123d.script.builders.scene_filter_builder import build_scene_filter
from py123d.script.run_conversion import build_worker
from py123d.script.utils.dataset_path_utils import setup_dataset_paths

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/viser"
CONFIG_NAME = "default_viser"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:

    setup_dataset_paths(cfg.dataset_paths)

    worker = build_worker(cfg)

    scene_filter = build_scene_filter(cfg.scene_filter)

    scene_builder = build_scene_builder(cfg.scene_builder)

    scenes = scene_builder.get_scenes(scene_filter, worker=worker)

    ViserViewer(scenes=scenes)


if __name__ == "__main__":
    main()
