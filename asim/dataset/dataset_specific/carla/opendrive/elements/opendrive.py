from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type
from xml.etree.ElementTree import Element, parse

import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString

LINE_STEP_SIZE: float = 0.1  # [m]


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
class PlanView:

    geometries: List[Geometry]

    @classmethod
    def parse(cls, plan_view_element: Optional[Element]) -> PlanView:

        args = {}
        geometries: List[Geometry] = []
        for geometry_element in plan_view_element.findall("geometry"):
            if geometry_element.find("line") is not None:
                geometry = Line.parse(geometry_element)
            elif geometry_element.find("arc") is not None:
                geometry = Arc.parse(geometry_element)
            else:
                warnings.warn(f"Warning..... Unknown geometry type {str(geometry_element)}")
                continue
            geometries.append(geometry)
        args["geometries"] = geometries
        return PlanView(**args)

    @property
    def linestring(self) -> LineString:
        return LineString(self.xy)

    @property
    def xy(self) -> npt.NDArray[np.float64]:

        pass


@dataclass
class Geometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float

    @property
    def linestring(self) -> LineString:
        return LineString(self.xy)

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError


@dataclass
class Line(Geometry):
    @classmethod
    def parse(cls, geometry_element: Element) -> Type[Geometry]:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        return cls(**args)

    @property
    def linestring(self) -> LineString:
        return LineString(self.xy)

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        s_line = np.arange(self.s, self.length + LINE_STEP_SIZE, LINE_STEP_SIZE, dtype=np.float64)
        s_line = np.clip(s_line, self.s, self.length)
        initial_point = np.array([self.x, self.y], dtype=np.float64)
        dxy = np.concatenate([s_line[..., None] * np.cos(self.hdg), s_line[..., None] * np.sin(self.hdg)], axis=-1)
        return initial_point + dxy


@dataclass
class Arc(Geometry):

    curvature: float

    @classmethod
    def parse(cls, geometry_element: Element) -> Type[Geometry]:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        args["curvature"] = float(geometry_element.find("arc").get("curvature"))
        return cls(**args)

    @property
    def linestring(self) -> LineString:
        return LineString(self.xy)

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        s_line = np.arange(self.s, self.length + LINE_STEP_SIZE, LINE_STEP_SIZE, dtype=np.float64)
        s_line = np.clip(s_line, self.s, self.length)

        # radius = 1.0 / self.curvature
        # hdg = self.hdg + s_line * self.curvature
        # x = self.x + radius * (np.sin(hdg) - np.sin(self.hdg))
        # y = self.y + radius * (np.cos(self.hdg) - np.cos(hdg))
        # return np.concatenate([x[..., None], y[..., None]], axis=-1)

        c = self.curvature
        hdg = self.hdg - np.pi / 2

        a = 2 / c * np.sin(s_line * c / 2)
        alpha = (np.pi - s_line * c) / 2 - hdg

        dx = -1 * a * np.cos(alpha)
        dy = a * np.sin(alpha)
        dxy = np.concatenate([dx[..., None], dy[..., None]], axis=-1)

        initial_point = np.array([self.x, self.y], dtype=np.float64)

        return initial_point + dxy


@dataclass
class Lanes:
    lane_offsets: List[LaneOffset]
    lane_sections: List[LaneSection]

    @classmethod
    def parse(cls, lanes_element: Optional[Element]) -> Lanes:
        args = {}
        lane_offsets: List[LaneOffset] = []
        for lane_offset_element in lanes_element.findall("laneOffset"):
            lane_offsets.append(LaneOffset.parse(lane_offset_element))
        args["lane_offsets"] = lane_offsets

        lane_sections: List[LaneSection] = []
        for lane_section_element in lanes_element.findall("laneSection"):
            lane_sections.append(LaneSection.parse(lane_section_element))
        args["lane_sections"] = lane_sections
        return Lanes(**args)


@dataclass
class LaneOffset:
    s: float = None
    type: str = None
    material: Optional[str] = None
    color: Optional[str] = None
    width: Optional[float] = None
    lane_change: Optional[str] = None

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        pass

    @classmethod
    def parse(cls, lane_offset_element: Optional[Element]) -> LaneOffset:
        args = {}
        args["s"] = float(lane_offset_element.get("s"))
        args["type"] = lane_offset_element.get("type")
        args["material"] = lane_offset_element.get("material")
        args["color"] = lane_offset_element.get("color")
        if lane_offset_element.get("width") is not None:
            args["width"] = float(lane_offset_element.get("width"))
        args["lane_change"] = lane_offset_element.get("lane_change")
        return LaneOffset(**args)


@dataclass
class LaneSection:
    s: float
    left: List[Lane]
    center: List[Lane]
    right: List[Lane]

    # def __post_init__(self):
    #     # NOTE: added assertion/filtering to check for element type or consistency
    #     pass

    @classmethod
    def parse(cls, lane_section_element: Optional[Element]) -> LaneSection:
        args = {}
        args["s"] = float(lane_section_element.get("s"))

        left: List[Lane] = []
        if lane_section_element.find("left") is not None:
            for lane_element in lane_section_element.find("left").findall("lane"):
                left.append(Lane.parse(lane_element))
        args["left"] = left

        center: List[Lane] = []
        if lane_section_element.find("center") is not None:
            for lane_element in lane_section_element.find("center").findall("lane"):
                center.append(Lane.parse(lane_element))
        args["center"] = center

        right: List[Lane] = []
        if lane_section_element.find("right") is not None:
            for lane_element in lane_section_element.find("right").findall("lane"):
                right.append(Lane.parse(lane_element))
        args["right"] = right

        return LaneSection(**args)


@dataclass
class Lane:

    id: int
    type: str
    level: bool

    widths: List[Width]
    # road_marks: List[RoadMarks]  # NOTE: not implemented

    predecessor: Optional[int] = None
    successor: Optional[int] = None

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        pass

    @classmethod
    def parse(cls, lane_element: Optional[Element]) -> Lane:
        # TODO: implement
        args = {}
        args["id"] = lane_element.get("id")
        args["type"] = lane_element.get("type")
        args["level"] = lane_element.get("level")

        if lane_element.find("link") is not None:
            if lane_element.find("link").find("predecessor") is not None:
                args["predecessor"] = int(lane_element.find("link").find("predecessor").get("id"))
            if lane_element.find("link").find("successor") is not None:
                args["successor"] = int(lane_element.find("link").find("successor").get("id"))

        widths: List[Width] = []
        for width_element in lane_element.findall("width"):
            widths.append(Width.parse(width_element))
        args["widths"] = widths

        return Lane(**args)


@dataclass
class Width:
    s_offset: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None

    @classmethod
    def parse(cls, width_element: Optional[Element]) -> Width:
        args = {}
        args["s_offset"] = float(width_element.get("sOffset"))
        args["a"] = float(width_element.get("a"))
        args["b"] = float(width_element.get("b"))
        args["c"] = float(width_element.get("c"))
        args["d"] = float(width_element.get("d"))
        return Width(**args)


@dataclass
class Junction:
    dummy: str = None

    @classmethod
    def parse(cls, junction_element: Optional[Element]) -> Junction:
        # TODO: implement

        return Junction()
