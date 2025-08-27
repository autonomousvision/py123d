from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property
from typing import Final, List, Optional, Union
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt

from d123.dataset.conversion.map.opendrive.parser.elevation import Elevation
from d123.dataset.conversion.map.opendrive.parser.geometry import Arc, Geometry, Line, Spiral
from d123.dataset.conversion.map.opendrive.parser.lane import LaneOffset, Width
from d123.dataset.conversion.map.opendrive.parser.polynomial import Polynomial
from d123.geometry import Point3DIndex, StateSE2Index

TOLERANCE: Final[float] = 1e-3


@dataclass
class PlanView:

    geometries: List[Geometry]

    def __post_init__(self):
        # Ensure geometries are sorted by their starting position 's'
        self.geometries.sort(key=lambda x: x.s)

    @classmethod
    def parse(cls, plan_view_element: Optional[Element]) -> PlanView:
        geometries: List[Geometry] = []
        for geometry_element in plan_view_element.findall("geometry"):
            if geometry_element.find("line") is not None:
                geometry = Line.parse(geometry_element)
            elif geometry_element.find("arc") is not None:
                geometry = Arc.parse(geometry_element)
            elif geometry_element.find("spiral") is not None:
                geometry = Spiral.parse(geometry_element)
            else:
                geometry_str = ET.tostring(geometry_element, encoding="unicode")
                raise NotImplementedError(f"Geometry not implemented: {geometry_str}")
            geometries.append(geometry)
        return PlanView(geometries=geometries)

    @cached_property
    def geometry_lengths(self) -> npt.NDArray[np.float64]:
        return np.cumsum([0.0] + [geo.length for geo in self.geometries], dtype=np.float64)

    @property
    def length(self) -> float:
        return float(self.geometry_lengths[-1])

    def interpolate_se2(self, s: float, t: float = 0.0, lane_section_end: bool = False) -> npt.NDArray[np.float64]:
        """
        Interpolates the SE2 state at a given longitudinal position s along the plan view.
        """
        if s > self.length:
            if np.isclose(s, self.length, atol=TOLERANCE):
                s = self.length
            else:
                raise ValueError(
                    f"PlanView: s={s} is beyond the end of the plan view (length={self.length}) with tolerance={TOLERANCE}."
                )

        # Find the geometry segment containing s
        geo_idx = np.searchsorted(self.geometry_lengths, s, side="right") - 1
        geo_idx = int(np.clip(geo_idx, 0, len(self.geometries) - 1))

        return self.geometries[geo_idx].interpolate_se2(s - self.geometry_lengths[geo_idx], t)


@dataclass
class ReferenceLine:

    reference_line: Union[ReferenceLine, PlanView]
    width_polynomials: List[Polynomial]
    elevations: List[Elevation]
    s_offset: float

    @property
    def length(self) -> float:
        return float(self.reference_line.length)

    @classmethod
    def from_plan_view(
        cls,
        plan_view: PlanView,
        lane_offsets: List[LaneOffset],
        elevations: List[Elevation],
    ) -> ReferenceLine:
        args = {}
        args["reference_line"] = plan_view
        args["width_polynomials"] = lane_offsets
        args["elevations"] = elevations
        args["s_offset"] = 0.0
        return ReferenceLine(**args)

    @classmethod
    def from_reference_line(
        cls,
        reference_line: ReferenceLine,
        widths: List[Width],
        s_offset: float = 0.0,
        t_sign: float = 1.0,
    ) -> ReferenceLine:
        assert t_sign in [1.0, -1.0], "t_sign must be either 1.0 or -1.0"

        args = {}
        args["reference_line"] = reference_line
        width_polynomials: List[Polynomial] = []
        for width in widths:
            width_polynomials.append(width.get_polynomial(t_sign=t_sign))
        args["width_polynomials"] = width_polynomials
        args["s_offset"] = s_offset
        args["elevations"] = reference_line.elevations

        return ReferenceLine(**args)

    @staticmethod
    def _find_polynomial(s: float, polynomials: List[Polynomial], lane_section_end: bool = False) -> Polynomial:

        out_polynomial = polynomials[-1]
        for polynomial in polynomials[::-1]:
            if lane_section_end:
                if polynomial.s < s:
                    out_polynomial = polynomial
                    break
            else:
                if polynomial.s <= s:
                    out_polynomial = polynomial
                    break

        # s_values = np.array([poly.s for poly in polynomials])
        # side = "left" if lane_section_end else "right"
        # poly_idx = np.searchsorted(s_values, s, side=side) - 1
        # poly_idx = int(np.clip(poly_idx, 0, len(polynomials) - 1))
        # return polynomials[poly_idx]
        return out_polynomial

    def interpolate_se2(self, s: float, t: float = 0.0, lane_section_end: bool = False) -> npt.NDArray[np.float64]:

        width_polynomial = self._find_polynomial(s, self.width_polynomials, lane_section_end=lane_section_end)
        t_offset = width_polynomial.get_value(s - width_polynomial.s)
        se2 = self.reference_line.interpolate_se2(self.s_offset + s, t=t_offset + t, lane_section_end=lane_section_end)

        return se2

    def interpolate_3d(self, s: float, t: float = 0.0, lane_section_end: bool = False) -> npt.NDArray[np.float64]:

        se2 = self.interpolate_se2(s, t, lane_section_end=lane_section_end)

        elevation_polynomial = self._find_polynomial(s, self.elevations, lane_section_end=lane_section_end)
        point_3d = np.zeros(len(Point3DIndex), dtype=np.float64)
        point_3d[Point3DIndex.XY] = se2[StateSE2Index.XY]
        point_3d[Point3DIndex.Z] = elevation_polynomial.get_value(s - elevation_polynomial.s)

        return point_3d
