import random
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional, Set, Union

from py123d.common.multithreading.worker_utils import WorkerPool, worker_map
from py123d.common.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.scene.abstract_scene import AbstractScene
from py123d.datatypes.scene.abstract_scene_builder import SceneBuilder
from py123d.datatypes.scene.arrow.arrow_scene import ArrowScene
from py123d.datatypes.scene.arrow.utils.arrow_metadata_utils import get_log_metadata_from_arrow
from py123d.datatypes.scene.scene_filter import SceneFilter
from py123d.datatypes.scene.scene_metadata import SceneExtractionMetadata


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
        for log_path in (dataset_path / "logs" / split_name).iterdir():
            if log_path.is_file() and log_path.name.endswith(".arrow"):
                if log_names is None or log_path.stem in log_names:
                    log_paths.append(log_path)
    return log_paths


def _extract_scenes_from_logs(log_paths: List[Path], filter: SceneFilter) -> List[AbstractScene]:
    scenes: List[AbstractScene] = []
    for log_path in log_paths:
        try:
            scene_extraction_metadatas = _get_scene_extraction_metadatas(log_path, filter)
        except Exception as e:
            print(f"Error extracting scenes from {log_path}: {e}")
            continue
        for scene_extraction_metadata in scene_extraction_metadatas:
            scenes.append(
                ArrowScene(
                    arrow_file_path=log_path,
                    scene_extraction_metadata=scene_extraction_metadata,
                )
            )
    return scenes


def _get_scene_extraction_metadatas(log_path: Union[str, Path], filter: SceneFilter) -> List[SceneExtractionMetadata]:
    scene_extraction_metadatas: List[SceneExtractionMetadata] = []

    recording_table = get_lru_cached_arrow_table(log_path)
    log_metadata = get_log_metadata_from_arrow(log_path)

    # 1. Filter location
    if (
        filter.locations is not None
        and log_metadata.map_metadata is not None
        and log_metadata.map_metadata.location not in filter.locations
    ):
        return scene_extraction_metadatas

    start_idx = int(filter.history_s / log_metadata.timestep_seconds) if filter.history_s is not None else 0
    end_idx = (
        len(recording_table) - int(filter.duration_s / log_metadata.timestep_seconds)
        if filter.duration_s is not None
        else len(recording_table)
    )
    if filter.duration_s is None:
        return [
            SceneExtractionMetadata(
                initial_uuid=str(recording_table["uuid"][start_idx].as_py()),
                initial_idx=start_idx,
                duration_s=(end_idx - start_idx) * log_metadata.timestep_seconds,
                history_s=filter.history_s if filter.history_s is not None else 0.0,
                iteration_duration_s=log_metadata.timestep_seconds,
            )
        ]

    scene_uuid_set = set(filter.scene_uuids) if filter.scene_uuids is not None else None

    for idx in range(start_idx, end_idx):
        scene_extraction_metadata: Optional[SceneExtractionMetadata] = None

        if scene_uuid_set is None:
            scene_extraction_metadata = SceneExtractionMetadata(
                initial_uuid=str(recording_table["uuid"][idx].as_py()),
                initial_idx=idx,
                duration_s=filter.duration_s,
                history_s=filter.history_s,
                iteration_duration_s=log_metadata.timestep_seconds,
            )
        elif str(recording_table["uuid"][idx]) in scene_uuid_set:
            scene_extraction_metadata = SceneExtractionMetadata(
                initial_uuid=str(recording_table["uuid"][idx].as_py()),
                initial_idx=idx,
                duration_s=filter.duration_s,
                history_s=filter.history_s,
                iteration_duration_s=log_metadata.timestep_seconds,
            )

        if scene_extraction_metadata is not None:
            # Check of timestamp threshold exceeded between previous scene, if specified in filter
            if filter.timestamp_threshold_s is not None and len(scene_extraction_metadatas) > 0:
                iteration_delta = idx - scene_extraction_metadatas[-1].initial_idx
                if (iteration_delta * log_metadata.timestep_seconds) < filter.timestamp_threshold_s:
                    continue

            # Check if camera data is available for the scene, if specified in filter
            # NOTE: We only check camera availability at the initial index of the scene.
            if filter.camera_types is not None:
                cameras_available = [
                    recording_table[f"{camera_type.serialize()}_data"][start_idx].as_py() is not None
                    for camera_type in filter.camera_types
                ]
                if not all(cameras_available):
                    continue

            scene_extraction_metadatas.append(scene_extraction_metadata)

    del recording_table, log_metadata
    return scene_extraction_metadatas
