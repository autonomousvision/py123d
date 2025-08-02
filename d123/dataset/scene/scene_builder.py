import abc
import json
import random
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional, Set, Union

from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

from d123.dataset.arrow.helper import open_arrow_table
from d123.dataset.logs.log_metadata import LogMetadata
from d123.dataset.scene.abstract_scene import AbstractScene
from d123.dataset.scene.arrow_scene import ArrowScene, SceneExtractionInfo
from d123.dataset.scene.scene_filter import SceneFilter

# TODO: Fix lazy abstraction implementation for scene builder.


class SceneBuilder(abc.ABC):
    @abc.abstractmethod
    def get_scenes(self, filter: SceneFilter, worker: WorkerPool) -> Iterator[AbstractScene]:
        """
        Returns an iterator over scenes that match the given filter.
        :param filter: SceneFilter object to filter the scenes.
        :param worker: WorkerPool to parallelize the scene extraction.
        :return: Iterator over AbstractScene objects.
        """
        raise NotImplementedError


class ArrowSceneBuilder(SceneBuilder):
    """
    A class to build a scene from a dataset.
    """

    def __init__(self, dataset_path: Union[str, Path]):
        self._dataset_path = Path(dataset_path)

    def get_scenes(self, filter: SceneFilter, worker: WorkerPool) -> Iterator[AbstractScene]:
        """See superclass."""

        split_types = set(filter.split_types) if filter.split_types else {"train", "val", "test"}
        split_names = (
            set(filter.split_names) if filter.split_names else _discover_split_names(self._dataset_path, split_types)
        )
        filter_log_names = set(filter.log_names) if filter.log_names else None
        log_paths = _discover_log_paths(self._dataset_path, split_names, filter_log_names)
        if len(log_paths) == 0:
            return []
        scenes = worker_map(worker, partial(_extract_scenes_from_logs, filter=filter), log_paths)

        if filter.shuffle:
            random.shuffle(scenes)

        if filter.max_num_scenes is not None:
            scenes = scenes[: filter.max_num_scenes]

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
        scene_extraction_infos = _get_scene_extraction_info(log_path, filter)
        for scene_extraction_info in scene_extraction_infos:
            scenes.append(
                ArrowScene(
                    arrow_file_path=log_path,
                    scene_extraction_info=scene_extraction_info,
                )
            )
    return scenes


def _get_scene_extraction_info(log_path: Union[str, Path], filter: SceneFilter) -> List[SceneExtractionInfo]:
    scene_extraction_infos: List[SceneExtractionInfo] = []

    recording_table = open_arrow_table(log_path)
    log_metadata = LogMetadata(**json.loads(recording_table.schema.metadata[b"log_metadata"].decode()))

    # 1. Filter map name
    if filter.map_names is not None and log_metadata.map_name not in filter.map_names:
        return scene_extraction_infos

    # 2. Filter by camera type if specified in filter
    if filter.camera_types is not None:
        if not all(camera_type.serialize() in recording_table.column_names for camera_type in filter.camera_types):
            return scene_extraction_infos

    start_idx = int(filter.history_s / log_metadata.timestep_seconds)  # if filter.history_s is not None else 0
    end_idx = (
        len(recording_table) - int(filter.duration_s / log_metadata.timestep_seconds)
        if filter.duration_s is not None
        else len(recording_table)
    )
    if filter.duration_s is None:
        return [
            SceneExtractionInfo(
                initial_idx=start_idx,
                duration_s=(end_idx - start_idx) * log_metadata.timestep_seconds,
                history_s=filter.history_s if filter.history_s is not None else 0.0,
                iteration_duration_s=log_metadata.timestep_seconds,
            )
        ]

    scene_token_set = set(filter.scene_tokens) if filter.scene_tokens is not None else None

    for idx in range(start_idx, end_idx):
        scene_extraction_info: Optional[SceneExtractionInfo] = None

        if scene_token_set is None:
            scene_extraction_info = SceneExtractionInfo(
                initial_idx=idx,
                duration_s=filter.duration_s,
                history_s=filter.history_s,
                iteration_duration_s=log_metadata.timestep_seconds,
            )
        elif str(recording_table["token"][idx]) in scene_token_set:
            scene_extraction_info = SceneExtractionInfo(
                initial_idx=idx,
                duration_s=filter.duration_s,
                history_s=filter.history_s,
                iteration_duration_s=log_metadata.timestep_seconds,
            )

        if scene_extraction_info is not None:
            # TODO: add more options
            if filter.timestamp_threshold_s is not None and len(scene_extraction_infos) > 0:
                iteration_delta = idx - scene_extraction_infos[-1].initial_idx
                if (iteration_delta * log_metadata.timestep_seconds) < filter.timestamp_threshold_s:
                    continue

            scene_extraction_infos.append(scene_extraction_info)

    del recording_table, log_metadata
    return scene_extraction_infos
