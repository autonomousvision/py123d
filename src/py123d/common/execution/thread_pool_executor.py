"""
Thread pool executor backend.
Code is adapted from the nuplan-devkit: https://github.com/motional/nuplan-devkit
"""

import logging
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from typing import Any, Iterable, List, Optional

from tqdm import tqdm

from py123d.common.execution.executor import (
    Executor,
    ExecutorResources,
    Task,
    get_max_size_of_arguments,
)

logger = logging.getLogger(__name__)


class ThreadPoolExecutor(Executor):
    """
    Distributes tasks across multiple threads on a single machine.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Create executor with a thread pool.
        :param max_workers: number of threads to use. Defaults to all available CPUs.
        """
        number_of_cpus_per_node = max_workers if max_workers else ExecutorResources.current_node_cpu_count()

        super().__init__(
            ExecutorResources(
                number_of_nodes=1, number_of_cpus_per_node=number_of_cpus_per_node, number_of_gpus_per_node=0
            )
        )

        self._executor = _ThreadPoolExecutor(max_workers=number_of_cpus_per_node)

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        return list(
            tqdm(
                self._executor.map(task.fn, *item_lists),
                leave=False,
                total=get_max_size_of_arguments(*item_lists),
                desc="ThreadPoolExecutor",
                disable=not verbose,
            )
        )

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        return self._executor.submit(task.fn, *args, **kwargs)
