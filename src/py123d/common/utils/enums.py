from __future__ import annotations

import enum

from pyparsing import Union


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class SerialIntEnum(enum.Enum):

    def __int__(self) -> int:
        return self.value

    def serialize(self, lower: bool = True) -> str:
        """Serialize the type when saving."""
        # Allow for lower/upper case letters during serialize
        return self.name.lower() if lower else self.name

    @classmethod
    def deserialize(cls, key: str) -> SerialIntEnum:
        """Deserialize the type when loading from a string."""
        # Allow for lower/upper case letters during deserialize
        return cls.__members__[key.upper()] if key.islower() else cls.__members__[key]

    @classmethod
    def from_int(cls, value: int) -> SerialIntEnum:
        """Get the enum from an int."""
        return cls(value)

    @classmethod
    def from_arbitrary(cls, value: Union[int, str, SerialIntEnum]) -> SerialIntEnum:
        """Get the enum from an int, string, or enum instance."""
        if isinstance(value, cls):
            return value
        elif isinstance(value, int):
            return cls.from_int(value)
        elif isinstance(value, str):
            return cls.deserialize(value)
        else:
            raise ValueError(f"Invalid value for {cls.__name__}: {value}")
