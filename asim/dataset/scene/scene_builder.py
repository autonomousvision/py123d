from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional, Set, Union

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

from asim.dataset.dataset_specific.nuplan.nuplan_data_processor import worker_map
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.dataset.scene.arrow_scene import ArrowScene
from asim.dataset.scene.scene_filter import SceneFilter


class ArrowSceneBuilder:
    """
    A class to build a scene from a dataset.
    """

    def __init__(self, dataset_path: Union[str, Path]):
        self._dataset_path = Path(dataset_path)

    def get_scenes(self, filter: SceneFilter, worker: WorkerPool) -> Iterator[AbstractScene]:

        split_types = set(filter.split_types) if filter.split_types else {"train", "val", "test"}
        split_names = (
            set(filter.split_names) if filter.split_names else _discover_split_names(self._dataset_path, split_types)
        )
        filter_log_names = set(filter.log_names) if filter.log_names else None
        log_paths = _discover_log_paths(self._dataset_path, split_names, filter_log_names)
        if len(log_paths) == 0:
            return iter([])
        scenes = worker_map(worker, partial(_extract_scenes_from_logs, filter=filter), log_paths)
        return scenes


def _discover_split_names(dataset_path: Path, split_types: Set[str]) -> Set[str]:
    assert set(split_types).issubset(
        {"train", "val", "test"}
    ), f"Invalid split types: {split_types}. Valid split types are 'train', 'val', 'test'."
    split_names: List[str] = []
    for split in dataset_path.iterdir():
        split_name = split.name
        if split.is_dir() and split.name != "maps":
            if any(split_type in split_name for split_type in split_types):
                split_names.append(split_name)

    return split_names


def _discover_log_paths(dataset_path: Path, split_names: Set[str], log_names: Optional[List[str]]) -> List[Path]:
    log_paths: List[Path] = []
    for split_name in split_names:
        for log_path in (dataset_path / split_name).iterdir():
            if log_path.is_file() and log_path.name.endswith(".arrow"):
                if log_names is None or log_path.stem in log_names:
                    log_paths.append(log_path)
    return log_paths


def _extract_scenes_from_logs(log_paths: List[Path], filter: SceneFilter) -> List[AbstractScene]:
    scenes: List[AbstractScene] = []
    for log_path in log_paths:
        scenes.append(ArrowScene(log_path))
    return scenes
