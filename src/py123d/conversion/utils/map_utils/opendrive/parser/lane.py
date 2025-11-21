from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element

from py123d.conversion.utils.map_utils.opendrive.parser.polynomial import XODRPolynomial


@dataclass
class XODRLanes:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/11_lanes/11_01_introduction.html
    """

    lane_offsets: List[XODRLaneOffset]
    lane_sections: List[XODRLaneSection]

    def __post_init__(self):
        self.lane_offsets.sort(key=lambda x: x.s, reverse=False)
        self.lane_sections.sort(key=lambda x: x.s, reverse=False)

    @classmethod
    def parse(cls, lanes_element: Optional[Element]) -> XODRLanes:
        args = {}
        lane_offsets: List[XODRLaneOffset] = []
        for lane_offset_element in lanes_element.findall("laneOffset"):
            lane_offsets.append(XODRLaneOffset.parse(lane_offset_element))
        args["lane_offsets"] = lane_offsets

        lane_sections: List[XODRLaneSection] = []
        for lane_section_element in lanes_element.findall("laneSection"):
            lane_sections.append(XODRLaneSection.parse(lane_section_element))
        args["lane_sections"] = lane_sections
        return XODRLanes(**args)

    @property
    def num_lane_sections(self) -> int:
        return len(self.lane_sections)

    @property
    def last_lane_section_idx(self) -> int:
        return self.num_lane_sections - 1


@dataclass
class XODRLaneOffset(XODRPolynomial):
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/11_lanes/11_04_lane_offset.html

    offset (ds) = a + b*ds + c*ds² + d*ds³
    """


@dataclass
class XODRLaneSection:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/11_lanes/11_03_lane_sections.html
    """

    s: float
    left_lanes: List[XODRLane]
    center_lanes: List[XODRLane]
    right_lanes: List[XODRLane]

    def __post_init__(self):
        self.left_lanes.sort(key=lambda x: x.id, reverse=False)
        self.right_lanes.sort(key=lambda x: x.id, reverse=True)
        # NOTE: added assertion/filtering to check for element type or consistency

    @classmethod
    def parse(cls, lane_section_element: Optional[Element]) -> XODRLaneSection:
        args = {}
        args["s"] = float(lane_section_element.get("s"))

        left_lanes: List[XODRLane] = []
        if lane_section_element.find("left") is not None:
            for lane_element in lane_section_element.find("left").findall("lane"):
                left_lanes.append(XODRLane.parse(lane_element))
        args["left_lanes"] = left_lanes

        center_lanes: List[XODRLane] = []
        if lane_section_element.find("center") is not None:
            for lane_element in lane_section_element.find("center").findall("lane"):
                center_lanes.append(XODRLane.parse(lane_element))
        args["center_lanes"] = center_lanes

        right_lanes: List[XODRLane] = []
        if lane_section_element.find("right") is not None:
            for lane_element in lane_section_element.find("right").findall("lane"):
                right_lanes.append(XODRLane.parse(lane_element))
        args["right_lanes"] = right_lanes

        return XODRLaneSection(**args)


@dataclass
class XODRLane:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/11_lanes/11_05_lane_link.html
    """

    id: int
    type: str
    level: bool

    widths: List[XODRWidth]
    road_marks: List[XODRRoadMark]

    predecessor: Optional[int] = None
    successor: Optional[int] = None

    def __post_init__(self):
        self.widths.sort(key=lambda x: x.s_offset, reverse=False)
        # NOTE: added assertion/filtering to check for element type or consistency

    @classmethod
    def parse(cls, lane_element: Optional[Element]) -> XODRLane:
        args = {}
        args["id"] = int(lane_element.get("id"))
        args["type"] = lane_element.get("type")
        args["level"] = lane_element.get("level")

        if lane_element.find("link") is not None:
            if lane_element.find("link").find("predecessor") is not None:
                args["predecessor"] = int(lane_element.find("link").find("predecessor").get("id"))
            if lane_element.find("link").find("successor") is not None:
                args["successor"] = int(lane_element.find("link").find("successor").get("id"))

        widths: List[XODRWidth] = []
        for width_element in lane_element.findall("width"):
            widths.append(XODRWidth.parse(width_element))
        args["widths"] = widths

        road_marks: List[XODRWidth] = []
        for road_mark_element in lane_element.findall("roadMark"):
            road_marks.append(XODRRoadMark.parse(road_mark_element))
        args["road_marks"] = road_marks

        return XODRLane(**args)


@dataclass
class XODRWidth:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/11_lanes/11_06_lane_geometry.html#sec-8d8ac2e0-b3d6-4048-a9ed-d5191af5c74b
    """

    s_offset: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None

    @classmethod
    def parse(cls, width_element: Optional[Element]) -> XODRWidth:
        args = {}
        args["s_offset"] = float(width_element.get("sOffset"))
        args["a"] = float(width_element.get("a"))
        args["b"] = float(width_element.get("b"))
        args["c"] = float(width_element.get("c"))
        args["d"] = float(width_element.get("d"))
        return XODRWidth(**args)

    def get_polynomial(self, t_sign: float = 1.0) -> XODRPolynomial:
        """
        Returns the polynomial representation of the width.
        """
        return XODRPolynomial(
            s=self.s_offset,
            a=self.a * t_sign,
            b=self.b * t_sign,
            c=self.c * t_sign,
            d=self.d * t_sign,
        )


@dataclass
class XODRRoadMark:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/11_lanes/11_08_road_markings.html
    """

    s_offset: float = None
    type: str = None
    material: Optional[str] = None
    color: Optional[str] = None
    width: Optional[float] = None
    lane_change: Optional[str] = None

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        pass

    @classmethod
    def parse(cls, road_mark_element: Optional[Element]) -> XODRRoadMark:
        args = {}
        args["s_offset"] = float(road_mark_element.get("sOffset"))
        args["type"] = road_mark_element.get("type")
        args["material"] = road_mark_element.get("material")
        args["color"] = road_mark_element.get("color")
        if road_mark_element.get("width") is not None:
            args["width"] = float(road_mark_element.get("width"))
        args["lane_change"] = road_mark_element.get("lane_change")
        return XODRRoadMark(**args)
