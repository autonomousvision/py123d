from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element

from py123d.conversion.utils.map_utils.opendrive.parser.polynomial import XODRPolynomial


@dataclass
class XORDElevationProfile:
    """
    Models elevation along s-axis of reference line.

    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_05_elevation.html#sec-1d876c00-d69e-46d9-bbcd-709ab48f14b1
    """

    elevations: List[XODRElevation]

    def __post_init__(self):
        self.elevations.sort(key=lambda x: x.s, reverse=False)

    @classmethod
    def parse(cls, elevation_profile_element: Optional[Element]) -> XORDElevationProfile:
        args = {}
        elevations: List[XODRElevation] = []
        if elevation_profile_element is not None:
            for elevation_element in elevation_profile_element.findall("elevation"):
                elevations.append(XODRElevation.parse(elevation_element))
        args["elevations"] = elevations
        return XORDElevationProfile(**args)


@dataclass
class XODRElevation(XODRPolynomial):
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_05_elevation.html#sec-66ac2b58-dc5e-4538-884d-204406ea53f2

    Represents an elevation profile in OpenDRIVE.
    """


@dataclass
class XODRLateralProfile:
    """
    Models elevation along t-axis of reference line.

    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_05_elevation.html#sec-66ac2b58-dc5e-4538-884d-204406ea53f2
    """

    super_elevations: List[XODRSuperElevation]
    shapes: List[XODRShape]

    def __post_init__(self):
        self.super_elevations.sort(key=lambda x: x.s, reverse=False)
        self.shapes.sort(key=lambda x: x.s, reverse=False)

    @classmethod
    def parse(cls, lateral_profile_element: Optional[Element]) -> XODRLateralProfile:
        args = {}

        super_elevations: List[XODRSuperElevation] = []
        shapes: List[XODRShape] = []

        if lateral_profile_element is not None:
            for super_elevation_element in lateral_profile_element.findall("superelevation"):
                super_elevations.append(XODRSuperElevation.parse(super_elevation_element))
            for shape_element in lateral_profile_element.findall("shape"):
                shapes.append(XODRShape.parse(shape_element))

        args["super_elevations"] = super_elevations
        args["shapes"] = shapes

        return XODRLateralProfile(**args)


@dataclass
class XODRSuperElevation(XODRPolynomial):
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/10_roads/10_05_elevation.html#sec-4abf7baf-fb2f-4263-8133-ad0f64f0feac

    superelevation (ds) = a + b*ds + c*ds² + d*ds³
    """


@dataclass
class XODRShape:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/1.8.0/specification/10_roads/10_05_elevation.html#sec-66ac2b58-dc5e-4538-884d-204406ea53f2

    hShape (dt)= a + b*dt + c*dt² + d*dt³
    """

    s: float
    t: float
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def parse(cls, shape_element: Element) -> XODRShape:
        args = {key: float(shape_element.get(key)) for key in ["s", "t", "a", "b", "c", "d"]}
        return XODRShape(**args)

    def get_value(self, dt: float) -> float:
        """
        Evaluate the polynomial at a given t value.
        """
        return self.a + self.b * dt + self.c * dt**2 + self.d * dt**3
