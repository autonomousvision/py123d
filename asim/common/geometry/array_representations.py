from enum import IntEnum


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class SE2Index(IntEnum):

    X = 0
    Y = 1
    HEADING = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)
