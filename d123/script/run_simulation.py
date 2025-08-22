import logging
import traceback
from pathlib import Path
from typing import Dict, List

import hydra
import lightning as L
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from d123.common.multithreading.worker_utils import worker_map
from d123.dataset.scene.abstract_scene import AbstractScene
from d123.script.builders.scene_builder_builder import build_scene_builder
from d123.script.builders.scene_filter_builder import build_scene_filter
from d123.script.run_dataset_conversion import build_worker
from d123.simulation.gym.demo_gym_env import DemoGymEnv
from d123.simulation.metrics.sim_agents.sim_agents import get_sim_agents_metrics

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

    results = worker_map(worker, _run_simulation, scenes)

    df = pd.DataFrame(results)
    avg_row = df.drop(columns=["token"]).mean(numeric_only=True)
    avg_row["token"] = "average"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    output_dir = Path(cfg.output_dir)
    df.to_csv(output_dir / "sim_agent_results.csv")


def _run_simulation(scenes: List[AbstractScene]) -> List[Dict[str, float]]:

    action = [1.0, 0.1]  # Placeholder action, replace with actual action logic
    env = DemoGymEnv(scenes)

    results = []

    for scene in tqdm(scenes):
        try:

            agent_rollouts = []

            map_api, ego_state, detection_observation, current_scene = env.reset(scene)
            agent_rollouts.append(detection_observation.box_detections)

            result = {}
            result["token"] = scene.token
            for i in range(150):
                ego_state, detection_observation, end = env.step(action)
                agent_rollouts.append(detection_observation.box_detections)
                if end:
                    break
            result.update(get_sim_agents_metrics(current_scene, agent_rollouts))
            results.append(result)
        except Exception:
            print(current_scene.token)
            traceback.print_exc()
            continue

        scene.close()
    return results


if __name__ == "__main__":
    main()
