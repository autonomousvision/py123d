from __future__ import annotations

from enum import IntEnum


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class SerialIntEnum(IntEnum):
    def serialize(self, lower: bool = True) -> str:
        """Serialize the type when saving."""
        # Allow for lower/upper case letters during serialize
        return self.name.lower() if lower else self.name

    @classmethod
    def deserialize(cls, key: str) -> type[SerialIntEnum]:
        """Deserialize the type when loading from a string."""
        # Allow for lower/upper case letters during deserialize
        return cls.__members__[key.upper()] if key.islower() else cls.__members__[key]

    @classmethod
    def from_int(cls, value: int) -> SerialIntEnum:
        """Get the enum from an int."""
        return cls(value)
