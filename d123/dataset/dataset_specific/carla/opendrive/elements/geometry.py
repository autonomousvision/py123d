from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt
from scipy.special import fresnel

from d123.common.geometry.base import StateSE2Index


@dataclass
class Geometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float

    @property
    def start_se2(self) -> npt.NDArray[np.float64]:
        start_se2 = np.zeros(len(StateSE2Index), dtype=np.float64)
        start_se2[StateSE2Index.X] = self.x
        start_se2[StateSE2Index.Y] = self.y
        start_se2[StateSE2Index.YAW] = self.hdg
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

        interpolated_se2 = self.start_se2.copy()
        interpolated_se2[StateSE2Index.X] += s * np.cos(self.hdg)
        interpolated_se2[StateSE2Index.Y] += s * np.sin(self.hdg)

        if t != 0.0:
            interpolated_se2[StateSE2Index.X] += t * np.cos(interpolated_se2[StateSE2Index.YAW] + np.pi / 2)
            interpolated_se2[StateSE2Index.Y] += t * np.sin(interpolated_se2[StateSE2Index.YAW] + np.pi / 2)

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

        interpolated_se2 = self.start_se2.copy()
        interpolated_se2[StateSE2Index.X] += dx
        interpolated_se2[StateSE2Index.Y] += dy
        interpolated_se2[StateSE2Index.YAW] += s * self.curvature

        if t != 0.0:
            interpolated_se2[StateSE2Index.X] += t * np.cos(interpolated_se2[StateSE2Index.YAW] + np.pi / 2)
            interpolated_se2[StateSE2Index.Y] += t * np.sin(interpolated_se2[StateSE2Index.YAW] + np.pi / 2)

        return interpolated_se2


@dataclass
class Geometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float

    @property
    def start_se2(self) -> npt.NDArray[np.float64]:
        start_se2 = np.zeros(len(StateSE2Index), dtype=np.float64)
        start_se2[StateSE2Index.X] = self.x
        start_se2[StateSE2Index.Y] = self.y
        start_se2[StateSE2Index.YAW] = self.hdg
        return start_se2

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        raise NotImplementedError


@dataclass
class Spiral(Geometry):

    curvature_start: float
    curvature_end: float

    @classmethod
    def parse(cls, geometry_element: Element) -> Geometry:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        spiral_element = geometry_element.find("spiral")
        args["curvature_start"] = float(spiral_element.get("curvStart"))
        args["curvature_end"] = float(spiral_element.get("curvEnd"))
        return cls(**args)

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        """
        https://en.wikipedia.org/wiki/Euler_spiral
        :param s: _description_
        :param t: _description_, defaults to 0.0
        :return: _description_
        """
        interpolated_se2 = self.start_se2.copy()

        gamma = (self.curvature_end - self.curvature_start) / self.length
        if abs(gamma) < 1e-10:
            print(gamma)
        # NOTE: doesn't consider case where gamma == 0

        dx, dy = self._compute_spiral_position(s, gamma)

        interpolated_se2[StateSE2Index.X] += dx
        interpolated_se2[StateSE2Index.Y] += dy
        interpolated_se2[StateSE2Index.YAW] += gamma * s**2 / 2 + self.curvature_start * s

        if t != 0.0:
            interpolated_se2[StateSE2Index.X] += t * np.cos(interpolated_se2[StateSE2Index.YAW] + np.pi / 2)
            interpolated_se2[StateSE2Index.Y] += t * np.sin(interpolated_se2[StateSE2Index.YAW] + np.pi / 2)

        return interpolated_se2

    def _compute_spiral_position(self, s: float, gamma: float) -> Tuple[float, float]:

        # Transform to normalized Fresnel spiral parameter
        # Standard Fresnel spiral has κ(u) = u, so we need to scale
        # Our spiral: κ(s) = κ₀ + γs
        # Standard: κ(u) = u

        # Use transformation: u = sqrt(2γ) * s + κ₀/sqrt(2γ)
        sqrt_2gamma = (2 * abs(gamma)) ** 0.5

        if gamma > 0:
            u_start = self.curvature_start / sqrt_2gamma
            u_end = u_start + sqrt_2gamma * s
        else:
            u_start = self.curvature_start / sqrt_2gamma
            u_end = u_start - sqrt_2gamma * s

        # Compute Fresnel integrals
        S_start, C_start = fresnel(u_start)
        S_end, C_end = fresnel(u_end)

        # Scale and rotate to our coordinate system
        scale_factor = 1.0 / sqrt_2gamma if gamma > 0 else -1.0 / sqrt_2gamma

        dC = C_end - C_start
        dS = S_end - S_start

        # Apply rotation by initial heading
        cos_hdg = np.cos(self.hdg)
        sin_hdg = np.sin(self.hdg)

        dx = scale_factor * (dC * cos_hdg - dS * sin_hdg)
        dy = scale_factor * (dC * sin_hdg + dS * cos_hdg)

        return dx, dy
