from __future__ import annotations

from asim.common.utils.enums import SerialIntEnum


class AgentType(SerialIntEnum):
    """
    Enum for agents in asim.
    """

    EGO = 0
    VEHICLE = 1
    PEDESTRIAN = 2
