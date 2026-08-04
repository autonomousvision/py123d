from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.arrow.modalities.sync_utils import get_sync_table
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution import Executor, ThreadPoolExecutor
from py123d.datatypes import Timestamp


def get_filtered_scenes(
    scene_filter: SceneFilter,
    data_root: Optional[Union[str, Path]] = None,
    executor: Executor = ThreadPoolExecutor(),
) -> List[SceneAPI]:
    """Retrieve a list of scenes that match the given filter criteria.

    :param scene_filter: Filter class describing criteria for scene selection.
    :param data_root: Root directory for py123d data, defaults to None
    :param executor: Executor for parallel execution, defaults to ThreadPoolExecutor()
    :return: List of scenes matching the filter criteria
    """

    if data_root is not None:
        data_root = Path(data_root)

    scenes = ArrowSceneBuilder(
        logs_root=data_root / "logs" if data_root is not None else None,
        maps_root=data_root / "maps" if data_root is not None else None,
    ).get_scenes(filter=scene_filter, executor=executor)

    return scenes


def get_scene_anchor_timestamps(scenes: List[SceneAPI]) -> List[Timestamp]:
    """Read the anchor (iteration-0) timestamps of many scenes at once.

    Equivalent to ``[scene.get_timestamp_at_iteration(0) for scene in scenes]``, but reads each log's
    sync-table timestamp column once instead of one row per scene, which is orders of magnitude
    faster over large scene lists. Non-Arrow scenes fall back to the per-scene read.

    :param scenes: Scenes to read the anchor timestamps of.
    :return: One timestamp per scene, in input order.
    """
    timestamp_column_by_log_dir: Dict[Path, np.ndarray] = {}
    anchor_timestamps: List[Timestamp] = []
    for scene in scenes:
        if not isinstance(scene, ArrowSceneAPI):
            anchor_timestamps.append(scene.get_timestamp_at_iteration(0))
            continue
        timestamp_column = timestamp_column_by_log_dir.get(scene.log_dir)
        if timestamp_column is None:
            timestamp_column = get_sync_table(scene.log_dir)["sync.timestamp_us"].to_numpy(zero_copy_only=False)
            timestamp_column_by_log_dir[scene.log_dir] = timestamp_column
        anchor_idx = scene.get_scene_metadata().initial_idx
        anchor_timestamps.append(Timestamp.from_us(int(timestamp_column[anchor_idx])))
    return anchor_timestamps
