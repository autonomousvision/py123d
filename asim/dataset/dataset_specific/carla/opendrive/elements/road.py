from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element

from asim.dataset.dataset_specific.carla.opendrive.elements.elevation import ElevationProfile, LateralProfile
from asim.dataset.dataset_specific.carla.opendrive.elements.lane import Lanes
from asim.dataset.dataset_specific.carla.opendrive.elements.objects import Object
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import PlanView


@dataclass
class Road:
    id: int
    junction: Optional[str]
    length: float  # [m]
    name: Optional[str]

    link: Link
    road_types: List[RoadType]
    plan_view: PlanView
    elevation_profile: ElevationProfile
    lateral_profile: LateralProfile
    lanes: Lanes
    objects: List[Object]

    rule: Optional[str] = None  # NOTE: ignored

    def __post_init__(self):
        self.rule = (
            "RHT" if self.rule is None else self.rule
        )  # FIXME: Find out the purpose RHT=right-hand traffic, LHT=left-hand traffic

    @classmethod
    def parse(cls, road_element: Element) -> Road:
        args = {}

        args["id"] = int(road_element.get("id"))
        args["junction"] = road_element.get("junction") if road_element.get("junction") != "-1" else None
        args["length"] = float(road_element.get("length"))
        args["name"] = road_element.get("name")

        args["link"] = Link.parse(road_element.find("link"))

        road_types: List[RoadType] = []
        for road_type_element in road_element.findall("type"):
            road_types.append(RoadType.parse(road_type_element))
        args["road_types"] = road_types

        args["plan_view"] = PlanView.parse(road_element.find("planView"))
        args["elevation_profile"] = ElevationProfile.parse(road_element.find("elevationProfile"))
        args["lateral_profile"] = LateralProfile.parse(road_element.find("lateralProfile"))

        args["lanes"] = Lanes.parse(road_element.find("lanes"))

        objects: List[Object] = []
        if road_element.find("objects") is not None:
            for object_element in road_element.find("objects").findall("object"):
                objects.append(Object.parse(object_element))
        args["objects"] = objects

        return Road(**args)


@dataclass
class Link:
    """Section 8.2"""

    predecessor: Optional[PredecessorSuccessor] = None
    successor: Optional[PredecessorSuccessor] = None

    @classmethod
    def parse(cls, link_element: Optional[Element]) -> PlanView:
        args = {}
        if link_element is not None:
            if link_element.find("predecessor") is not None:
                args["predecessor"] = PredecessorSuccessor.parse(link_element.find("predecessor"))
            if link_element.find("successor") is not None:
                args["successor"] = PredecessorSuccessor.parse(link_element.find("successor"))
        return Link(**args)


@dataclass
class PredecessorSuccessor:
    element_type: Optional[str] = None
    element_id: Optional[int] = None
    contact_point: Optional[str] = None

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        assert self.contact_point is None or self.contact_point in ["start", "end"]

    @classmethod
    def parse(cls, element: Element) -> PredecessorSuccessor:
        args = {}
        args["element_type"] = element.get("elementType")
        args["element_id"] = int(element.get("elementId"))
        args["contact_point"] = element.get("contactPoint")
        return PredecessorSuccessor(**args)


@dataclass
class RoadType:

    s: Optional[float] = None
    type: Optional[str] = None
    speed: Optional[Speed] = None

    @classmethod
    def parse(cls, road_type_element: Optional[Element]) -> RoadType:
        args = {}
        if road_type_element is not None:
            args["s"] = float(road_type_element.get("s"))
            args["type"] = road_type_element.get("type")
            args["speed"] = Speed.parse(road_type_element.find("speed"))
        return RoadType(**args)


@dataclass
class Speed:
    max: Optional[float] = None
    unit: Optional[str] = None

    @classmethod
    def parse(cls, speed_element: Optional[Element]) -> Speed:
        args = {}
        if speed_element is not None:
            args["max"] = float(speed_element.get("max"))
            args["unit"] = speed_element.get("unit")
        return Speed(**args)
