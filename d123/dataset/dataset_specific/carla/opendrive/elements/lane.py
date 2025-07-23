from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt


@dataclass
class Lanes:
    lane_offsets: List[LaneOffset]
    lane_sections: List[LaneSection]

    def __post_init__(self):
        self.lane_offsets.sort(key=lambda x: x.s, reverse=False)
        self.lane_sections.sort(key=lambda x: x.s, reverse=False)

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

    @property
    def num_lane_sections(self):
        return len(self.lane_sections)

    @property
    def last_lane_section_idx(self):
        return self.num_lane_sections - 1


@dataclass
class LaneOffset:
    """Section 11.4"""

    s: float
    a: float
    b: float
    c: float
    d: float

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        pass

    @classmethod
    def parse(cls, lane_offset_element: Element) -> LaneOffset:
        args = {key: float(lane_offset_element.get(key)) for key in ["s", "a", "b", "c", "d"]}
        return LaneOffset(**args)

    @property
    def polynomial_coefficients(self) -> npt.NDArray[np.float64]:
        return np.array([self.a, self.b, self.c, self.d], dtype=np.float64)


@dataclass
class LaneSection:
    s: float
    left_lanes: List[Lane]
    center_lanes: List[Lane]
    right_lanes: List[Lane]

    def __post_init__(self):
        self.left_lanes.sort(key=lambda x: x.id, reverse=False)
        self.right_lanes.sort(key=lambda x: x.id, reverse=True)
        # NOTE: added assertion/filtering to check for element type or consistency

    @classmethod
    def parse(cls, lane_section_element: Optional[Element]) -> LaneSection:
        args = {}
        args["s"] = float(lane_section_element.get("s"))

        left_lanes: List[Lane] = []
        if lane_section_element.find("left") is not None:
            for lane_element in lane_section_element.find("left").findall("lane"):
                left_lanes.append(Lane.parse(lane_element))
        args["left_lanes"] = left_lanes

        center_lanes: List[Lane] = []
        if lane_section_element.find("center") is not None:
            for lane_element in lane_section_element.find("center").findall("lane"):
                center_lanes.append(Lane.parse(lane_element))
        args["center_lanes"] = center_lanes

        right_lanes: List[Lane] = []
        if lane_section_element.find("right") is not None:
            for lane_element in lane_section_element.find("right").findall("lane"):
                right_lanes.append(Lane.parse(lane_element))
        args["right_lanes"] = right_lanes

        return LaneSection(**args)


@dataclass
class Lane:

    id: int
    type: str
    level: bool

    widths: List[Width]
    road_marks: List[RoadMark]

    predecessor: Optional[int] = None
    successor: Optional[int] = None

    def __post_init__(self):
        self.widths.sort(key=lambda x: x.s_offset, reverse=False)
        # NOTE: added assertion/filtering to check for element type or consistency

    @classmethod
    def parse(cls, lane_element: Optional[Element]) -> Lane:
        args = {}
        args["id"] = int(lane_element.get("id"))
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

        road_marks: List[Width] = []
        for road_mark_element in lane_element.findall("roadMark"):
            road_marks.append(RoadMark.parse(road_mark_element))
        args["road_marks"] = road_marks

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

    @property
    def polynomial_coefficients(self) -> npt.NDArray[np.float64]:
        return np.array([self.a, self.b, self.c, self.d], dtype=np.float64)


@dataclass
class RoadMark:
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
    def parse(cls, road_mark_element: Optional[Element]) -> RoadMark:
        args = {}
        args["s_offset"] = float(road_mark_element.get("sOffset"))
        args["type"] = road_mark_element.get("type")
        args["material"] = road_mark_element.get("material")
        args["color"] = road_mark_element.get("color")
        if road_mark_element.get("width") is not None:
            args["width"] = float(road_mark_element.get("width"))
        args["lane_change"] = road_mark_element.get("lane_change")
        return RoadMark(**args)
