from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt

from asim.dataset.dataset_specific.carla.opendrive.elements.lane import Lanes
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import PlanView


@dataclass
class Road:
    id: int
    junction: Optional[str]
    length: float  # [m]
    name: Optional[str]

    link: Link
    road_type: RoadType
    plan_view: PlanView
    elevation_profile: ElevationProfile
    lateral_profile: LateralProfile
    lanes: Lanes

    rule: Optional[str] = None  # NOTE: ignored

    def __post_init__(self):
        self.rule = (
            "RHT" if self.rule is None else self.rule
        )  # FIXME: Find out the purpose RHT=right-hand traffic, LHT=left-hand traffic

    @classmethod
    def parse(cls, road_element: Element) -> Road:
        # TODO: implement
        args = {}
        # try:

        args["id"] = int(road_element.get("id"))
        args["junction"] = road_element.get("junction") if road_element.get("junction") != "-1" else None
        args["length"] = float(road_element.get("length"))
        args["name"] = road_element.get("name")

        args["link"] = Link.parse(road_element.find("link"))
        args["road_type"] = RoadType.parse(road_element.find("type"))
        args["plan_view"] = PlanView.parse(road_element.find("planView"))
        args["elevation_profile"] = ElevationProfile.parse(road_element.find("elevationProfile"))
        args["lateral_profile"] = LateralProfile.parse(road_element.find("lateralProfile"))

        args["lanes"] = Lanes.parse(road_element.find("lanes"))
        # except:
        #     road_name = road_element.get("name")
        #     print(f"Failure in road {road_name}")

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
    def parse(cls, speed_element: Optional[Element]) -> RoadType:
        args = {}
        if speed_element is not None:
            args["max"] = float(speed_element.get("max"))
            args["unit"] = speed_element.get("unit")
        return Speed(**args)


@dataclass
class ElevationProfile:
    elevations: List[Elevation]

    def __post_init__(self):
        self.elevations.sort(key=lambda x: x.s, reverse=False)

    @classmethod
    def parse(cls, elevation_profile_element: Optional[Element]) -> ElevationProfile:
        args = {}
        elevations: List[Elevation] = []
        if elevation_profile_element is not None:
            for elevation_element in elevation_profile_element.findall("elevation"):
                elevations.append(Elevation.parse(elevation_element))
        args["elevations"] = elevations
        return ElevationProfile(**args)


@dataclass
class Elevation:
    """TODO: Refactor and merge with other elements, e.g. LaneOffset"""

    s: float
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def parse(cls, elevation_element: Element) -> Elevation:
        args = {key: float(elevation_element.get(key)) for key in ["s", "a", "b", "c", "d"]}
        return Elevation(**args)

    @property
    def polynomial_coefficients(self) -> npt.NDArray[np.float64]:
        return np.array([self.a, self.b, self.c, self.d], dtype=np.float64)


@dataclass
class LateralProfile:

    superelevations: List[SuperElevation]
    shapes: List[Shape]

    def __post_init__(self):
        self.superelevations.sort(key=lambda x: x.s, reverse=False)
        self.shapes.sort(key=lambda x: x.s, reverse=False)

    @classmethod
    def parse(cls, lateral_profile_element: Optional[Element]) -> LateralProfile:
        args = {}

        superelevations: List[SuperElevation] = []
        shapes: List[Shape] = []

        if lateral_profile_element is not None:
            for superelevation_element in lateral_profile_element.findall("superelevation"):
                superelevations.append(SuperElevation.parse(superelevation_element))
            for shape_element in lateral_profile_element.findall("shape"):
                shapes.append(Shape.parse(shape_element))

        args["superelevations"] = superelevations
        args["shapes"] = shapes

        return LateralProfile(**args)


@dataclass
class SuperElevation:
    """TODO: Refactor and merge with other elements, e.g. Elevation, LaneOffset"""

    s: float
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def parse(cls, super_elevation_element: Element) -> SuperElevation:
        args = {key: float(super_elevation_element.get(key)) for key in ["s", "a", "b", "c", "d"]}
        return SuperElevation(**args)

    @property
    def polynomial_coefficients(self) -> npt.NDArray[np.float64]:
        return np.array([self.a, self.b, self.c, self.d], dtype=np.float64)


@dataclass
class Shape:
    """TODO: Refactor and merge with other elements, e.g. Elevation, LaneOffset"""

    s: float
    t: float
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def parse(cls, shape_element: Element) -> Shape:
        args = {key: float(shape_element.get(key)) for key in ["s", "t", "a", "b", "c", "d"]}
        return Shape(**args)

    @property
    def polynomial_coefficients(self) -> npt.NDArray[np.float64]:
        return np.array([self.a, self.b, self.c, self.d], dtype=np.float64)
