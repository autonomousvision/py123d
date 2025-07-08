from __future__ import annotations

from dataclasses import dataclass

# TODO: Refactor
# NOTE: Taken from nuplan and adapted. Generally, these types are handy, when handling time discrete data.


class TimeDuration:
    """Class representing a time delta, with a microsecond resolution."""

    __slots__ = "_time_us"

    def __init__(self, *, time_us: int, _direct: bool = True) -> None:
        """Constructor, should not be called directly. Raises if the keyword parameter _direct is not set to false."""
        if _direct:
            raise RuntimeError("Don't initialize this class directly, use one of the constructors instead!")

        self._time_us = time_us

    @classmethod
    def from_us(cls, t_us: int) -> TimeDuration:
        """
        Constructs a TimeDuration from a value in microseconds.
        :param t_us: Time in microseconds.
        :return: TimeDuration.
        """
        assert isinstance(t_us, int), "Microseconds must be an integer!"
        return cls(time_us=t_us, _direct=False)

    @classmethod
    def from_ms(cls, t_ms: float) -> TimeDuration:
        """
        Constructs a TimeDuration from a value in milliseconds.
        :param t_ms: Time in milliseconds.
        :return: TimeDuration.
        """
        return cls(time_us=int(t_ms * int(1e3)), _direct=False)

    @classmethod
    def from_s(cls, t_s: float) -> TimeDuration:
        """
        Constructs a TimeDuration from a value in seconds.
        :param t_s: Time in seconds.
        :return: TimeDuration.
        """
        return cls(time_us=int(t_s * int(1e6)), _direct=False)

    @property
    def time_us(self) -> int:
        """
        :return: TimeDuration in microseconds.
        """
        return self._time_us

    @property
    def time_ms(self) -> float:
        """
        :return: TimeDuration in milliseconds.
        """
        return self._time_us / 1e3

    @property
    def time_s(self) -> float:
        """
        :return: TimeDuration in seconds.
        """
        return self._time_us / 1e6

    def __add__(self, other: object) -> TimeDuration:
        """
        Adds a time duration to a time duration.
        :param other: time duration.
        :return: self + other if other is a TimeDuration.
        """
        if isinstance(other, TimeDuration):
            return TimeDuration.from_us(self.time_us + other.time_us)
        return NotImplemented

    def __sub__(self, other: object) -> TimeDuration:
        """
        Subtract a time duration from a time duration.
        :param other: time duration.
        :return: self - other if other is a TimeDuration.
        """
        if isinstance(other, TimeDuration):
            return TimeDuration.from_us(self.time_us - other.time_us)
        return NotImplemented

    def __mul__(self, other: object) -> TimeDuration:
        """
        Multiply a time duration by a scalar value.
        :param other: value to multiply.
        :return: self * other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s * other)
        return NotImplemented

    def __rmul__(self, other: object) -> TimeDuration:
        """
        Multiply a time duration by a scalar value.
        :param other: value to multiply.
        :return: self * other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    def __truediv__(self, other: object) -> TimeDuration:
        """
        Divides a time duration by a scalar value.
        :param other: value to divide for.
        :return: self / other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s / other)
        return NotImplemented

    def __floordiv__(self, other: object) -> TimeDuration:
        """
        Floor divides a time duration by a scalar value.
        :param other: value to divide for.
        :return: self // other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s // other)
        return NotImplemented

    def __gt__(self, other: TimeDuration) -> bool:
        """
        Self is greater than other.
        :param other: TimeDuration.
        :return: True if self > other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us > other.time_us
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """
        Self is greater or equal than other.
        :param other: TimeDuration.
        :return: True if self >= other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us >= other.time_us
        return NotImplemented

    def __lt__(self, other: TimeDuration) -> bool:
        """
        Self is less than other.
        :param other: TimeDuration.
        :return: True if self < other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us < other.time_us
        return NotImplemented

    def __le__(self, other: TimeDuration) -> bool:
        """
        Self is less or equal than other.
        :param other: TimeDuration.
        :return: True if self <= other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us <= other.time_us
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """
        Self is equal to other.
        :param other: TimeDuration.
        :return: True if self == other, False otherwise.
        """
        if not isinstance(other, TimeDuration):
            return NotImplemented

        return self.time_us == other.time_us

    def __hash__(self) -> int:
        """
        :return: hash for this object.
        """
        return hash(self.time_us)

    def __repr__(self) -> str:
        """
        :return: String representation.
        """
        return "TimeDuration({}s)".format(self.time_s)


@dataclass
class TimePoint:
    """
    Time instance in a time series.
    """

    time_us: int  # [micro seconds] time since epoch in micro seconds
    __slots__ = "time_us"

    def __post_init__(self) -> None:
        """
        Validate class after creation.
        """
        assert self.time_us >= 0, "Time point has to be positive!"

    @classmethod
    def from_us(cls, t_us: int) -> TimePoint:
        """
        Constructs a TimePoint from a value in microseconds.
        :param t_us: Time in microseconds.
        :return: TimePoint.
        """
        assert isinstance(t_us, int), "Microseconds must be an integer!"
        return TimePoint(time_us=t_us)

    @classmethod
    def from_ms(cls, t_ms: float) -> TimePoint:
        """
        Constructs a TimePoint from a value in milliseconds.
        :param t_ms: Time in milliseconds.
        :return: TimePoint.
        """
        return TimePoint(time_us=int(t_ms * int(1e3)))

    @classmethod
    def from_s(cls, t_s: float) -> TimePoint:
        """
        Constructs a TimePoint from a value in seconds.
        :param t_s: Time in seconds.
        :return: TimePoint.
        """
        return TimePoint(time_us=int(t_s * int(1e6)))

    @property
    def time_ms(self) -> float:
        """
        :return: TimePoint in milliseconds.
        """
        return self.time_us / 1e3

    @property
    def time_s(self) -> float:
        """
        :return: TimePoint in seconds.
        """
        return self.time_us / 1e6

    def __add__(self, other: object) -> TimePoint:
        """
        Adds a TimeDuration to generate a new TimePoint.
        :param other: time point.
        :return: self + other.
        """
        if isinstance(other, (TimeDuration, TimePoint)):
            return TimePoint(self.time_us + other.time_us)
        return NotImplemented

    def __radd__(self, other: object) -> TimePoint:
        """
        :param other: Right addition target.
        :return: Addition with other if other is a TimeDuration.
        """
        if isinstance(other, TimeDuration):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other: object) -> TimePoint:
        """
        Subtract a time duration from a time point.
        :param other: time duration.
        :return: self - other if other is a TimeDuration.
        """
        if isinstance(other, (TimeDuration, TimePoint)):
            return TimePoint(self.time_us - other.time_us)
        return NotImplemented

    def __gt__(self, other: TimePoint) -> bool:
        """
        Self is greater than other.
        :param other: time point.
        :return: True if self > other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us > other.time_us
        return NotImplemented

    def __ge__(self, other: TimePoint) -> bool:
        """
        Self is greater or equal than other.
        :param other: time point.
        :return: True if self >= other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us >= other.time_us
        return NotImplemented

    def __lt__(self, other: TimePoint) -> bool:
        """
        Self is less than other.
        :param other: time point.
        :return: True if self < other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us < other.time_us
        return NotImplemented

    def __le__(self, other: TimePoint) -> bool:
        """
        Self is less or equal than other.
        :param other: time point.
        :return: True if self <= other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us <= other.time_us
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """
        Self is equal to other
        :param other: time point
        :return: True if self == other, False otherwise
        """
        if not isinstance(other, TimePoint):
            return NotImplemented

        return self.time_us == other.time_us

    def __hash__(self) -> int:
        """
        :return: hash for this object
        """
        return hash(self.time_us)

    def diff(self, time_point: TimePoint) -> TimeDuration:
        """
        Computes the TimeDuration between self and another TimePoint.
        :param time_point: The other time point.
        :return: The TimeDuration between the two TimePoints.
        """
        return TimeDuration.from_us(int(self.time_us - time_point.time_us))
