from __future__ import annotations

import numpy as np


class Timestamp:
    """Timestamp class representing a time point in microseconds."""

    __slots__ = ("_time_us",)
    _time_us: int  # [micro seconds] time since epoch in micro seconds

    @classmethod
    def from_ns(cls, t_ns: int) -> Timestamp:
        """Constructs a Timestamp from a value in nanoseconds.

        :param t_ns: Time in nanoseconds.
        :return: Timestamp.
        """
        assert isinstance(t_ns, (int, np.integer)), "Nanoseconds must be an integer!"
        instance = object.__new__(cls)
        setattr(instance, "_time_us", t_ns // 1000)
        return instance

    @classmethod
    def from_us(cls, t_us: int) -> Timestamp:
        """Constructs a Timestamp from a value in microseconds.

        :param t_us: Time in microseconds.
        :return: Timestamp.
        """
        assert isinstance(t_us, (int, np.integer)), f"Microseconds must be an integer, got {type(t_us)}!"
        instance = object.__new__(cls)
        setattr(instance, "_time_us", t_us)
        return instance

    @classmethod
    def from_ms(cls, t_ms: float) -> Timestamp:
        """Constructs a Timestamp from a value in milliseconds.

        :param t_ms: Time in milliseconds.
        :return: Timestamp.
        """
        instance = object.__new__(cls)
        setattr(instance, "_time_us", int(t_ms * int(1e3)))
        return instance

    @classmethod
    def from_s(cls, t_s: float) -> Timestamp:
        """Constructs a Timestamp from a value in seconds.

        :param t_s: Time in seconds.
        :return: Timestamp.
        """
        instance = object.__new__(cls)
        setattr(instance, "_time_us", int(t_s * int(1e6)))
        return instance

    @property
    def time_ns(self) -> int:
        """The timestamp in nanoseconds [ns]."""
        return self._time_us * 1000

    @property
    def time_us(self) -> int:
        """The timestamp in microseconds [μs]."""
        return self._time_us

    @property
    def time_ms(self) -> float:
        """The timestamp in milliseconds [ms]."""
        return self._time_us / 1e3

    @property
    def time_s(self) -> float:
        """The timestamp in seconds [s]."""
        return self._time_us / 1e6

    def __repr__(self):
        """String representation of :class:`Timestamp`."""
        return f"Timestamp(time_us={self._time_us})"
