import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Timer:
    """
    A simple timer class to measure execution time of different parts of the code.
    """

    def __init__(self, name: Optional[str] = None, end_key: str = "total"):
        """
        Initializes the Timer instance.
        :param name: Name of the Timer, defaults to None
        :param end_key: name of the final row, defaults to "total"
        """

        self._name = name
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
        """Called during the start of the timer ."""
        self._start_time = time.perf_counter()
        self._iteration_time = time.perf_counter()

    def log(self, key: str) -> None:
        """
        Called after code block execution. Logs the time taken for the block, given the name (key).
        :param key: Name of the code block to log the time for.
        """
        if key not in self._time_logs.keys():
            self._time_logs[key] = []

        self._time_logs[key].append(time.perf_counter() - self._iteration_time)
        self._iteration_time = time.perf_counter()

    def end(self) -> None:
        """Called at the end of the timer."""
        if self._end_key not in self._time_logs.keys():
            self._time_logs[self._end_key] = []

        self._time_logs[self._end_key].append(time.perf_counter() - self._start_time)

    def stats(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with statistics of the logged times.
        :param verbose: whether to print the timings, defaults to True
        :return: pandas dataframe.F
        """

        statistics = {}

        for key, timings in self._time_logs.items():

            timings_array = np.array(timings)
            timings_statistics = {}

            for name, function in self._statistic_functions.items():
                timings_statistics[name] = function(timings_array)

            statistics[key] = timings_statistics

        dataframe = pd.DataFrame.from_dict(statistics).transpose()

        if verbose:
            print(dataframe.to_string())

        return dataframe

    def info(self) -> Dict[str, float]:
        """
        Summarized information about the timings.
        :return: Dictionary with the mean of each timing.
        """
        info = {}
        for key, timings in self._time_logs.items():
            info[key] = np.array(timings).mean()
        return info

    def flush(self) -> None:
        """Clears the logged times."""
        self._time_logs: Dict[str, List[float]] = {}
        self._start_time: Optional[float] = None
        self._iteration_time: Optional[float] = None
