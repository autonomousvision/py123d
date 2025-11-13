import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Timer:
    """Simple Timer class to log time taken by code blocks.

    Example
    -------
    >>> timer = Timer()
    >>> timer.start()
    >>> time.sleep(0.1)  # Simulate code block
    >>> timer.log("block_1")
    >>> time.sleep(0.2)  # Simulate another code block
    >>> timer.log("block_2")
    >>> timer.end()
    >>> print(timer) # Displays timing statistics (with some variation)
                mean       min       max     argmax    median
    block_1  0.100123  0.100123  0.100123         0  0.100123
    block_2  0.200456  0.200456  0.200456         0  0.200456
    total    0.300579  0.300579  0.300579         0  0.300579

    """

    def __init__(self, end_key: str = "total"):
        """Initializes the :class:`Timer`

        :param end_key: The key used to log the total time, defaults to "total"
        """
        self._end_key: str = end_key
        self._statistic_functions = {
            "mean": np.mean,
            "min": np.min,
            "max": np.max,
            "argmax": np.argmax,
            "median": np.median,
        }

        self._time_logs: Dict[str, List[float]] = {}

        self._start_time: Optional[float] = None
        self._iteration_time: Optional[float] = None

    def start(self) -> None:
        """Called at the start of the timer."""
        self._start_time = time.perf_counter()
        self._iteration_time = time.perf_counter()

    def log(self, key: str) -> None:
        """
        Called after code block execution. Logs the time taken for the block, given the name (key).
        :param key: Unique identifier of the code block to log the time for.
        """
        assert self._iteration_time is not None, "Timer has not been started. Call start() before logging."

        if key not in self._time_logs.keys():
            self._time_logs[key] = []

        self._time_logs[key].append(time.perf_counter() - self._iteration_time)
        self._iteration_time = time.perf_counter()

    def end(self) -> None:
        """Called at the end of the timer."""
        assert self._start_time is not None, "Timer has not been started. Call start() before logging."
        if self._end_key not in self._time_logs.keys():
            self._time_logs[self._end_key] = []

        self._time_logs[self._end_key].append(time.perf_counter() - self._start_time)

    def to_pandas(self) -> Optional[pd.DataFrame]:
        """Returns a DataFrame with statistics of the logged times.

        :return: pandas dataframe.
        """

        statistics = {}
        for key, timings in self._time_logs.items():
            timings_array = np.array(timings)
            timings_statistics = {}
            for name, function in self._statistic_functions.items():
                timings_statistics[name] = function(timings_array)
            statistics[key] = timings_statistics
        dataframe = pd.DataFrame.from_dict(statistics).transpose()

        return dataframe

    def info(self) -> Dict[str, float]:
        """Summarized information about the timings.

        :return: Dictionary with the mean of each timing.
        """
        info = {}
        for key, timings in self._time_logs.items():
            info[key] = {}
            for name, function in self._statistic_functions.items():
                info[key][name] = function(np.array(timings))
        return info

    def flush(self) -> None:
        """Clears the logged times."""
        self._time_logs: Dict[str, List[float]] = {}
        self._start_time: Optional[float] = None
        self._iteration_time: Optional[float] = None

    def __str__(self) -> str:
        """String representation of the Timer."""
        dataframe = self.to_pandas()
        return dataframe.to_string() if dataframe is not None else "No timings logged"
