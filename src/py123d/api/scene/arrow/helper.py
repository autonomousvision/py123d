from pathlib import Path
from typing import List, Optional, Union

from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.multithreading.worker_parallel import SingleMachineParallelExecutor
from py123d.common.multithreading.worker_pool import WorkerPool


def get_filtered_scenes(
    scene_filter: SceneFilter,
    py123d_data_root: Optional[Union[str, Path]] = None,
    worker: Optional[WorkerPool] = None,
) -> List[SceneAPI]:
    """Retrieve a list of scenes that match the given filter criteria.

    :param scene_filter: Filter class describing criteria for scene selection.
    :param py123d_data_root: Root directory for py123d data, defaults to None
    :param worker: Worker pool for parallel execution, defaults to None
    :return: List of scenes matching the filter criteria
    """

    if worker is None:
        worker = SingleMachineParallelExecutor()

    if py123d_data_root is not None:
        py123d_data_root = Path(py123d_data_root)

    scenes = ArrowSceneBuilder(
        logs_root=py123d_data_root / "logs" if py123d_data_root is not None else None,
        maps_root=py123d_data_root / "maps" if py123d_data_root is not None else None,
    ).get_scenes(filter=scene_filter, worker=worker)

    return scenes
