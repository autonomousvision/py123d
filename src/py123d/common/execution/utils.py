"""
Execution utilities.
Code is adapted from the nuplan-devkit: https://github.com/motional/nuplan-devkit
"""

from typing import Any, Callable, List, Optional

import numpy as np
from psutil import cpu_count

from py123d.common.execution.executor import Executor, Task


def chunk_list(input_list: List[Any], num_chunks: Optional[int] = None) -> List[List[Any]]:
    """
    Chunks a list to equal sized lists. The size of the last list might be truncated.
    :param input_list: List to be chunked.
    :param num_chunks: Number of chunks, equals to the number of cores if set to None.
    :return: List of equal sized lists.
    """
    num_chunks = num_chunks if num_chunks else cpu_count(logical=True)
    chunks = np.array_split(input_list, num_chunks)  # type: ignore
    return [chunk.tolist() for chunk in chunks if len(chunk) != 0]


def executor_map(executor: Executor, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
    """
    Map a list of objects through an executor.
    :param executor: Executor to use for parallelization.
    :param fn: Function to use when mapping.
    :param input_objects: List of objects to map.
    :return: List of mapped objects.
    """
    if executor.number_of_threads == 0:
        return fn(input_objects)

    object_chunks = chunk_list(input_objects, executor.number_of_threads)
    scattered_objects = executor.map(Task(fn=fn), object_chunks)
    output_objects = [result for results in scattered_objects for result in results]

    return output_objects
