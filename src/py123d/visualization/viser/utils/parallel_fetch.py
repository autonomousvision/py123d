"""Shared worker pool for per-sensor data fetching in viewer elements.

Each sensor stream lives in its own arrow file, so fetching/decoding different
sensors concurrently does not contend on a shared reader, and the heavy work
(image decode, display ISP, numpy transforms) releases the GIL.
"""

import concurrent.futures
import threading
from typing import Callable, Iterable, List, Optional, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")

_MAX_WORKERS = 8

_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def fetch_parallel(fn: Callable[[_T], _R], items: Iterable[_T]) -> List[_R]:
    """Apply fn to every item concurrently on a shared pool, preserving input order.

    Exceptions raised by fn propagate to the caller, matching sequential semantics.
    """
    items = list(items)
    if len(items) <= 1:
        return [fn(item) for item in items]

    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_MAX_WORKERS, thread_name_prefix="viser-element-fetch"
            )
    return list(_executor.map(fn, items))
