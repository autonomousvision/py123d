from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element

from py123d.conversion.utils.map_utils.opendrive.parser.elevation import XODRLateralProfile, XORDElevationProfile
from py123d.conversion.utils.map_utils.opendrive.parser.lane import XODRLanes
from py123d.conversion.utils.map_utils.opendrive.parser.objects import XODRObject
from py123d.conversion.utils.map_utils.opendrive.parser.reference import XODRPlanView


@dataclass
class XODRRoad:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_01_introduction.html
    """

    id: int
    junction: Optional[str]
    length: float
    name: Optional[str]

    link: XODRLink
    road_types: List[XODRRoadType]
    plan_view: XODRPlanView
    elevation_profile: XORDElevationProfile
    lateral_profile: XODRLateralProfile
    lanes: XODRLanes
    objects: List[XODRObject]

    rule: Optional[str] = None  # NOTE: ignored

    def __post_init__(self):
        self.rule = (
            "RHT" if self.rule is None else self.rule
        )  # FIXME: Find out the purpose RHT=right-hand traffic, LHT=left-hand traffic

    @classmethod
    def parse(cls, road_element: Element) -> XODRRoad:
        args = {}

        args["id"] = int(road_element.get("id"))
        args["junction"] = road_element.get("junction") if road_element.get("junction") != "-1" else None
        args["length"] = float(road_element.get("length"))
        args["name"] = road_element.get("name")

        args["link"] = XODRLink.parse(road_element.find("link"))

        road_types: List[XODRRoadType] = []
        for road_type_element in road_element.findall("type"):
            road_types.append(XODRRoadType.parse(road_type_element))
        args["road_types"] = road_types

        args["plan_view"] = XODRPlanView.parse(road_element.find("planView"))
        args["elevation_profile"] = XORDElevationProfile.parse(road_element.find("elevationProfile"))
        args["lateral_profile"] = XODRLateralProfile.parse(road_element.find("lateralProfile"))

        args["lanes"] = XODRLanes.parse(road_element.find("lanes"))

        objects: List[XODRObject] = []
        if road_element.find("objects") is not None:
            for object_element in road_element.find("objects").findall("object"):
                objects.append(XODRObject.parse(object_element))

        args["objects"] = objects

        return XODRRoad(**args)


@dataclass
class XODRLink:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_03_road_linkage.html
    """

    predecessor: Optional[XODRPredecessorSuccessor] = None
    successor: Optional[XODRPredecessorSuccessor] = None

    @classmethod
    def parse(cls, link_element: Optional[Element]) -> XODRPlanView:
        args = {}
        if link_element is not None:
            if link_element.find("predecessor") is not None:
                args["predecessor"] = XODRPredecessorSuccessor.parse(link_element.find("predecessor"))
            if link_element.find("successor") is not None:
                args["successor"] = XODRPredecessorSuccessor.parse(link_element.find("successor"))
        return XODRLink(**args)


@dataclass
class XODRPredecessorSuccessor:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_03_road_linkage.html
    """

    element_type: Optional[str] = None
    element_id: Optional[int] = None
    contact_point: Optional[str] = None

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        assert self.contact_point is None or self.contact_point in ["start", "end"]

    @classmethod
    def parse(cls, element: Element) -> XODRPredecessorSuccessor:
        args = {}
        args["element_type"] = element.get("elementType")
        args["element_id"] = int(element.get("elementId"))
        args["contact_point"] = element.get("contactPoint")
        return XODRPredecessorSuccessor(**args)


@dataclass
class XODRRoadType:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_04_road_type.html
    """

    s: Optional[float] = None
    type: Optional[str] = None
    speed: Optional[XODRSpeed] = None

    @classmethod
    def parse(cls, road_type_element: Optional[Element]) -> XODRRoadType:
        args = {}
        if road_type_element is not None:
            args["s"] = float(road_type_element.get("s"))
            args["type"] = road_type_element.get("type")
            args["speed"] = XODRSpeed.parse(road_type_element.find("speed"))
        return XODRRoadType(**args)


@dataclass
class XODRSpeed:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_04_road_type.html#sec-33dc6899-854e-4533-a3d9-76e9e1518ee7
    """

    max: Optional[float] = None
    unit: Optional[str] = None

    @classmethod
    def parse(cls, speed_element: Optional[Element]) -> XODRSpeed:
        args = {}
        if speed_element is not None:
            args["max"] = float(speed_element.get("max"))
            args["unit"] = speed_element.get("unit")
        return XODRSpeed(**args)
