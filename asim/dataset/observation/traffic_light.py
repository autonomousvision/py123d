from asim.common.utils.enums import SerialIntEnum


class TrafficLightStatusType(SerialIntEnum):
    """
    Enum for TrafficLightStatusType.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3
