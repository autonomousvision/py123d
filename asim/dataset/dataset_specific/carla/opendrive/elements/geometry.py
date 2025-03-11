from __future__ import annotations

from dataclasses import dataclass
from typing import Type
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString

LINE_STEP_SIZE = 0.1


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

    def get_xy(self, s: float) -> npt.NDArray[np.float64]:
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
        initial_point = np.array([self.x, self.y], dtype=np.float64)
        offset_point = np.array(
            [self.x + self.length * np.cos(self.hdg), self.y + self.length * np.sin(self.hdg)], dtype=np.float64
        )
        return np.concatenate([initial_point[None, ...], offset_point[None, ...]], axis=0, dtype=np.float64)

    def get_xy(self, s: float) -> npt.NDArray[np.float64]:
        x = self.x + s * np.cos(self.hdg)
        y = self.y + s * np.sin(self.hdg)
        return np.array([x, y], dtype=np.float64)


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
        s_line = np.linspace(0, self.length, num=int(self.length // LINE_STEP_SIZE), endpoint=True)

        radius = 1.0 / self.curvature
        hdg = self.hdg + s_line * self.curvature
        x = self.x + radius * (np.sin(hdg) - np.sin(self.hdg))
        y = self.y + radius * (np.cos(self.hdg) - np.cos(hdg))
        return np.concatenate([x[..., None], y[..., None]], axis=-1)

    def get_xy(self, s: float) -> npt.NDArray[np.float64]:
        print("arc")
        radius = 1.0 / self.curvature
        hdg = self.hdg + s * self.curvature
        x = self.x + radius * (np.sin(hdg) - np.sin(self.hdg))
        y = self.y + radius * (np.cos(self.hdg) - np.cos(hdg))
        return np.array([x, y], dtype=np.float64)
