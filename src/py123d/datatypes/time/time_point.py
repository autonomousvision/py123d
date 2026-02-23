from __future__ import annotations

import numpy as np


class TimePoint:
    """Time instance in a time series."""

    __slots__ = ("_time_us",)
    _time_us: int  # [micro seconds] time since epoch in micro seconds

    @classmethod
    def from_ns(cls, t_ns: int) -> TimePoint:
        """Constructs a TimePoint from a value in nanoseconds.

        :param t_ns: Time in nanoseconds.
        :return: TimePoint.
        """
        assert isinstance(t_ns, (int, np.integer)), "Nanoseconds must be an integer!"
        instance = object.__new__(cls)
        object.__setattr__(instance, "_time_us", t_ns // 1000)
        return instance

    @classmethod
    def from_us(cls, t_us: int) -> TimePoint:
        """Constructs a TimePoint from a value in microseconds.

        :param t_us: Time in microseconds.
        :return: TimePoint.
        """
        assert isinstance(t_us, (int, np.integer)), f"Microseconds must be an integer, got {type(t_us)}!"
        instance = object.__new__(cls)
        object.__setattr__(instance, "_time_us", t_us)
        return instance

    @classmethod
    def from_ms(cls, t_ms: float) -> TimePoint:
        """Constructs a TimePoint from a value in milliseconds.

        :param t_ms: Time in milliseconds.
        :return: TimePoint.
        """
        instance = object.__new__(cls)
        object.__setattr__(instance, "_time_us", int(t_ms * int(1e3)))
        return instance

    @classmethod
    def from_s(cls, t_s: float) -> TimePoint:
        """Constructs a TimePoint from a value in seconds.

        :param t_s: Time in seconds.
        :return: TimePoint.
        """
        instance = object.__new__(cls)
        object.__setattr__(instance, "_time_us", int(t_s * int(1e6)))
        return instance

    @property
    def time_ns(self) -> int:
        """The timepoint in nanoseconds [ns]."""
        return self._time_us * 1000

    @property
    def time_us(self) -> int:
        """The timepoint in microseconds [μs]."""
        return self._time_us

    @property
    def time_ms(self) -> float:
        """The timepoint in milliseconds [ms]."""
        return self._time_us / 1e3

    @property
    def time_s(self) -> float:
        """The timepoint in seconds [s]."""
        return self._time_us / 1e6

    def __repr__(self):
        """String representation of :class:`TimePoint`."""
        return f"TimePoint(time_us={self._time_us})"
