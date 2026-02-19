"""
Sequential executor backend.
Code is adapted from the nuplan-devkit: https://github.com/motional/nuplan-devkit
"""

import logging
from concurrent.futures import Future
from typing import Any, Iterable, List

from tqdm import tqdm

from py123d.common.execution.executor import (
    Executor,
    ExecutorResources,
    Task,
    get_max_size_of_arguments,
)

logger = logging.getLogger(__name__)


class SequentialExecutor(Executor):
    """
    Executes all tasks sequentially in the current process.
    """

    def __init__(self) -> None:
        """
        Initialize simple sequential executor.
        """
        super().__init__(ExecutorResources(number_of_nodes=1, number_of_cpus_per_node=1, number_of_gpus_per_node=0))

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        if task.num_cpus not in [None, 1]:
            raise ValueError(f"Expected num_cpus to be 1 or unset for SequentialExecutor, got {task.num_cpus}")
        output = [
            task.fn(*args)
            for args in tqdm(
                zip(*item_lists),
                leave=False,
                total=get_max_size_of_arguments(*item_lists),
                desc="SequentialExecutor",
                disable=not verbose,
            )
        ]
        return output

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        raise NotImplementedError
