from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt


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

    super_elevations: List[SuperElevation]
    shapes: List[Shape]

    def __post_init__(self):
        self.super_elevations.sort(key=lambda x: x.s, reverse=False)
        self.shapes.sort(key=lambda x: x.s, reverse=False)

    @classmethod
    def parse(cls, lateral_profile_element: Optional[Element]) -> LateralProfile:
        args = {}

        super_elevations: List[SuperElevation] = []
        shapes: List[Shape] = []

        if lateral_profile_element is not None:
            for super_elevation_element in lateral_profile_element.findall("superelevation"):
                super_elevations.append(SuperElevation.parse(super_elevation_element))
            for shape_element in lateral_profile_element.findall("shape"):
                shapes.append(Shape.parse(shape_element))

        args["super_elevations"] = super_elevations
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
