from enum import Enum, IntEnum


class NamedEnum(Enum):
    def __init__(self, index, names):
        self._index = index
        self._names = names

    @property
    def index(self):
        return self._index

    @property
    def names(self):
        return self._names

    @classmethod
    def from_name(cls, name):
        for member in cls:
            if name in member.names:
                return member
        raise ValueError(f"No enum member with name '{name}'")


class OpenDriveLaneType(IntEnum):

    BIKING = (0, ("biking",))
    BORDER = (1, ("border",))
    CONNECTING_RAMP = (2, ("connectingRamp",))
    CURB = (3, ("curb",))
    DRIVING = (4, ("driving",))
    ENTRY = (5, ("entry",))
    EXIT = (6, ("exit",))
    MEDIAN = (7, ("median",))
    NONE = (8, ("none",))
    OFF_RAMP = (9, ("offRamp",))
    ON_RAMP = (10, ("onRamp",))
    PARKING = (11, ("parking",))
    RAIL = (12, ("rail",))
    RESTRICTED = (13, ("restricted",))
    SHOULDER = (14, ("shoulder",))
    SIDEWALK = (15, ("sidewalk", "walking"))
    SLIP_LANE = (16, ("slipLane",))
    STOP = (17, ("stop",))
    TRAM = (18, ("tram",))
