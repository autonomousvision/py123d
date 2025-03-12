from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from xml.etree.ElementTree import Element, parse

from asim.dataset.dataset_specific.carla.opendrive.elements.lane import Lanes
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import PlanView

LINE_STEP_SIZE: float = 1  # [m]


@dataclass
class OpenDRIVE:

    header: Header

    roads: List[Road]
    # controllers: List[Controllers] = EMPTY_LIST_FIELD # NOTE: not implemented
    junctions: List[Junction]

    @classmethod
    def parse(cls, root_element: Element) -> OpenDRIVE:

        args = {}
        args["header"] = Header.parse(root_element.find("header"))

        roads: List[Road] = []
        for road_element in root_element.findall("road"):
            roads.append(Road.parse(road_element))
        args["roads"] = roads

        junctions: List[Junction] = []
        for junction_element in root_element.findall("junction"):
            junctions.append(Junction.parse(junction_element))
        args["junctions"] = junctions

        return OpenDRIVE(**args)

    @classmethod
    def parse_from_file(cls, file_path: Path) -> OpenDRIVE:
        tree = parse(file_path)
        return OpenDRIVE.parse(tree.getroot())


@dataclass
class Header:
    """Section 4.4.2"""

    rev_major: Optional[int] = None
    rev_minor: Optional[int] = None
    name: Optional[str] = None
    version: Optional[str] = None
    data: Optional[str] = None
    north: Optional[float] = None
    south: Optional[float] = None
    east: Optional[float] = None
    west: Optional[float] = None
    vendor: Optional[str] = None
    geo_reference: Optional[str] = None

    @classmethod
    def parse(cls, header_element: Optional[Element]) -> Header:
        """
        :param header_element: XML element containing the OpenDRIVE header.
        :return: instance of OpenDRIVE header dataclass.
        """
        args = {}
        if header_element is not None:
            args["rev_major"] = header_element.get("rev_major")
            args["rev_minor"] = header_element.get("rev_minor")
            args["name"] = header_element.get("name")
            args["version"] = header_element.get("version")
            args["data"] = header_element.get("data")
            args["north"] = float(header_element.get("north"))
            args["south"] = float(header_element.get("south"))
            args["east"] = float(header_element.get("east"))
            args["west"] = float(header_element.get("west"))
            args["vendor"] = header_element.get("vendor")
            if header_element.find("geoReference") is not None:
                args["geo_reference"] = header_element.find("geoReference").text

        return Header(**args)


@dataclass
class Road:
    id: str
    junction: Optional[str]
    length: float  # [m]
    name: Optional[str]

    link: Link
    road_type: RoadType
    plan_view: PlanView

    lanes: Lanes

    # elevation_profile: List[ElevationProfile] # NOTE: not implemented
    # lateral_profile: List[LateralProfile] # NOTE: not implemented
    rule: Optional[str] = None  # NOTE: ignored

    def __post_init__(self):
        self.rule = (
            "RHT" if self.rule is None else self.rule
        )  # FIXME: Find out the purpose RHT=right-hand traffic, LHT=left-hand traffic

    @classmethod
    def parse(cls, road_element: Element) -> Road:
        # TODO: implement
        args = {}

        args["id"] = road_element.get("id")
        args["junction"] = road_element.get("junction") if road_element.get("junction") != "-1" else None
        args["length"] = float(road_element.get("length"))
        args["name"] = road_element.get("name")

        args["road_type"] = RoadType.parse(road_element.find("type"))
        args["link"] = Link.parse(road_element.find("link"))
        args["plan_view"] = PlanView.parse(road_element.find("planView"))

        args["lanes"] = Lanes.parse(road_element.find("lanes"))

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
    element_id: Optional[str] = None
    contact_point: Optional[str] = None

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        pass

    @classmethod
    def parse(cls, element: Element) -> PredecessorSuccessor:
        args = {}
        args["element_type"] = element.get("elementType")
        args["element_id"] = element.get("elementId")
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
    def parse(cls, speed_element: Optional[Element]) -> RoadType:
        args = {}
        if speed_element is not None:
            args["max"] = float(speed_element.get("max"))
            args["unit"] = speed_element.get("unit")
        return Speed(**args)


@dataclass
class Junction:
    dummy: str = None

    @classmethod
    def parse(cls, junction_element: Optional[Element]) -> Junction:
        # TODO: implement

        return Junction()
