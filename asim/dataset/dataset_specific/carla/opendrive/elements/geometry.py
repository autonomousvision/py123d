from __future__ import annotations

from dataclasses import dataclass
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt

from asim.common.geometry.array_representations import SE2Index


@dataclass
class Geometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float

    @property
    def start_se2(self) -> npt.NDArray[np.float64]:
        start_se2 = np.zeros(len(SE2Index), dtype=np.float64)
        start_se2[SE2Index.X] = self.x
        start_se2[SE2Index.Y] = self.y
        start_se2[SE2Index.HEADING] = self.hdg
        return start_se2

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        raise NotImplementedError


@dataclass
class Line(Geometry):
    @classmethod
    def parse(cls, geometry_element: Element) -> Geometry:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        return cls(**args)

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        assert s >= 0.0, f"s = {s}"

        # s = np.clip(s, 0.0, self.length - self.s)

        interpolated_se2 = self.start_se2
        interpolated_se2[SE2Index.X] += s * np.cos(self.hdg)
        interpolated_se2[SE2Index.Y] += s * np.sin(self.hdg)

        if t != 0.0:
            pass

        return interpolated_se2


@dataclass
class Arc(Geometry):

    curvature: float

    @classmethod
    def parse(cls, geometry_element: Element) -> Geometry:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        args["curvature"] = float(geometry_element.find("arc").get("curvature"))
        return cls(**args)

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:

        c = self.curvature
        hdg = self.hdg - np.pi / 2

        a = 2 / c * np.sin(s * c / 2)
        alpha = (np.pi - s * c) / 2 - hdg

        dx = -1 * a * np.cos(alpha)
        dy = a * np.sin(alpha)

        interpolated_se2 = self.start_se2
        interpolated_se2[SE2Index.X] += dx
        interpolated_se2[SE2Index.Y] += dy
        interpolated_se2[SE2Index.Y] += s * self.curvature

        return interpolated_se2
