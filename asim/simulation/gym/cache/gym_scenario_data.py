from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, List


@dataclass
class GymScenarioData:
    """Helper data class to store scenario data in a lightweight format."""

    map: int

    time_points: List[int]
    past_time_points: List[int]
    route_roadblock_ids: List[str]

    ego_states: List[List[float]]
    past_ego_states: List[List[float]]

    tracked_objects: List[GymTrackedObjects]
    traffic_lights: List[GymTrafficLights]

    def __post_init__(self):
        time_lengths = [
            len(self.time_points),
            len(self.tracked_objects),
            len(self.traffic_lights),
        ]
        assert len(set(time_lengths)) == 1, f"{time_lengths}"

    @classmethod
    def deserialize(cls, dict: Dict[str, Any]) -> GymScenarioData:
        """
        Deserialize a dictionary into a GymScenarioData object.
        :param dict: stored dictionary containing data.
        :return: GymScenarioData object.
        """
        return GymScenarioData(
            map=dict["map"],
            time_points=dict["time_points"],
            past_time_points=dict["past_time_points"],
            route_roadblock_ids=dict["route_roadblock_ids"],
            ego_states=dict["ego_states"],
            past_ego_states=dict["past_ego_states"],
            tracked_objects=[GymTrackedObjects.deserialize(dict_) for dict_ in dict["tracked_objects"]],
            traffic_lights=[GymTrafficLights.deserialize(dict_) for dict_ in dict["traffic_lights"]],
        )


@dataclass
class GymTrackedObjects:
    """Helper data class to store tracked objects in a lightweight format."""

    states: List[List[float]]
    tracked_object_types: List[int]
    track_tokens: List[str]

    def __post_init__(self):
        time_lengths = [
            len(self.states),
            len(self.tracked_object_types),
            len(self.track_tokens),
        ]
        assert len(set(time_lengths)) == 1

    @classmethod
    def deserialize(cls, dict: Dict[str, Any]) -> GymTrackedObjects:
        return GymTrackedObjects(
            states=dict["states"],
            tracked_object_types=dict["tracked_object_types"],
            track_tokens=dict["track_tokens"],
        )


@dataclass
class GymTrafficLights:
    """Helper data class to store traffic light data in a lightweight format."""

    states: List[int]
    lane_connector_ids: List[int]

    def __post_init__(self):
        time_lengths = [
            len(self.states),
            len(self.lane_connector_ids),
        ]
        assert len(set(time_lengths)) == 1

    @classmethod
    def deserialize(cls, dict: Dict[str, Any]) -> GymTrafficLights:
        return GymTrafficLights(
            states=dict["states"],
            lane_connector_ids=dict["lane_connector_ids"],
        )


class TrackedObjectIndex(IntEnum):
    """Intenum for the indices of the tracked object state vector."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4
    _VELOCITY_X = 5
    _VELOCITY_Y = 6

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def VELOCITY_X(cls):
        return cls._VELOCITY_X

    @classmethod
    @property
    def VELOCITY_Y(cls):
        return cls._VELOCITY_Y

    @classmethod
    @property
    def POINT(cls):
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        return slice(cls._X, cls._HEADING + 1)

    @classmethod
    @property
    def VELOCITY(cls):
        return slice(cls._VELOCITY_X, cls._VELOCITY_Y + 1)


class MapEnum(Enum):
    """Enum of maps in nuPlan."""

    SINGAPORE = 0, "sg-one-north"
    BOSTON = 1, "us-ma-boston"
    LAS_VEGAS = 2, "us-nv-las-vegas-strip"
    PITTSBURGH = 3, "us-pa-pittsburgh-hazelwood"

    def __int__(self) -> int:
        """
        Convert an element to int
        :return: int
        """
        return self.value  # type: ignore

    def __new__(cls, value: int, name: str) -> MapEnum:
        """
        Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name  # type: ignore
        return member

    def __eq__(self, other: object) -> bool:
        """
        Equality checking
        :return: int
        """
        try:
            return self.name == other.name and self.value == other.value  # type: ignore
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash"""
        return hash((self.name, self.value))

    @classmethod
    def from_fullname(cls, fullname: str) -> MapEnum:
        """
        Get the enum instance from fullname
        :param fullname: the fullname of the map
        :return: MapEnum instance
        """
        for member in cls:
            if member.fullname == fullname:
                return member
        raise ValueError(f"No MapEnum member with fullname {fullname}")
