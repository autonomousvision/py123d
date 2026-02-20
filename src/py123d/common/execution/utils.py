"""
Execution utilities.
Code is adapted from the nuplan-devkit: https://github.com/motional/nuplan-devkit
"""

from typing import Any, Callable, List, Optional

import numpy as np
from psutil import cpu_count

from py123d.common.execution.executor import Executor, Task


def chunk_list(input_list: List[Any], num_chunks: Optional[int] = None) -> List[List[Any]]:
    """Chunks a list to equal sized lists. The size of the last list might be truncated.

    :param input_list: List to be chunked.
    :param num_chunks: Number of chunks, equals to the number of cores if set to None.
    :return: List of equal sized lists.
    """
    num_chunks = num_chunks if num_chunks else cpu_count(logical=True)
    chunks = np.array_split(input_list, num_chunks)  # type: ignore
    return [chunk.tolist() for chunk in chunks if len(chunk) != 0]


def executor_map_chunked_single(
    executor: Executor, fn: Callable[..., Any], input_objects: List[Any], name: Optional[str] = None
) -> List[Any]:
    """Map a function over individual objects through an executor. The input list is pre-chunked into
    equal-sized chunks (one per worker), and fn is called once per object within each chunk.

    :param executor: Executor to use for parallelization.
    :param fn: Function that takes a single object and returns a single result.
    :param input_objects: List of objects to map.
    :param name: Optional name for the progress bar description.
    :return: List of mapped objects.
    """

    def _chunked_fn(chunk: List[Any]) -> List[Any]:
        return [fn(obj) for obj in chunk]

    return executor_map_chunked_list(executor, _chunked_fn, input_objects, name=name)


def executor_map_chunked_list(
    executor: Executor, fn: Callable[..., List[Any]], input_objects: List[Any], name: Optional[str] = None
) -> List[Any]:
    """Map a list of objects through an executor. The input list is pre-chunked into
    equal-sized chunks (one per worker), and fn receives an entire chunk at once.

    :param executor: Executor to use for parallelization.
    :param fn: Function that takes a list of objects and returns a list of results.
    :param input_objects: List of objects to map.
    :param name: Optional name for the progress bar description.
    :return: List of mapped objects.
    """
    if executor.number_of_threads == 0:
        return fn(input_objects)

    object_chunks = chunk_list(input_objects, executor.number_of_threads)
    scattered_objects = executor.map(Task(fn=fn), object_chunks, desc=name)
    output_objects = [result for results in scattered_objects for result in results]

    return output_objects


def executor_map_queued(
    executor: Executor, fn: Callable[..., Any], input_objects: List[Any], name: Optional[str] = None
) -> List[Any]:
    """Map a function over individual objects through an executor without pre-chunking.
    Each item is submitted as an individual task, so workers dynamically pick up the next
    item when they become free. This provides better load balancing when tasks have
    varying execution times, at the cost of higher scheduling overhead.

    :param executor: Executor to use for parallelization.
    :param fn: Function that takes a single object and returns a single result.
    :param input_objects: List of objects to map.
    :param name: Optional name for the progress bar description.
    :return: List of mapped objects.
    """
    if executor.number_of_threads == 0:
        return [fn(obj) for obj in input_objects]

    return executor.map(Task(fn=fn), input_objects, desc=name)
