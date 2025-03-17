from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Union
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt

from asim.common.geometry.base_enum import StateSE2Index
from asim.common.geometry.utils import normalize_angle
from asim.dataset.dataset_specific.carla.opendrive.elements.geometry import Arc, Geometry, Line
from asim.dataset.dataset_specific.carla.opendrive.elements.lane import LaneOffset


@dataclass
class PlanView:

    geometries: List[Geometry]

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        self.geometries.sort(key=lambda x: x.s, reverse=False)

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

    @cached_property
    def geometry_lengths(self) -> npt.NDArray[np.float64]:
        return np.cumsum([0.0] + [geo.length for geo in self.geometries], dtype=np.float64)

    @property
    def length(self) -> float:
        return float(self.geometry_lengths[-1])

    def interpolate_se2(self, s: float, t: float = 0.0, is_last_pos: bool = False) -> npt.NDArray[np.float64]:

        try:
            # get index of geometry which is at s_pos
            mask = self.geometry_lengths > s
            sub_idx = np.argmin(self.geometry_lengths[mask] - s)
            geo_idx = np.arange(self.geometry_lengths.shape[0])[mask][sub_idx] - 1
        except ValueError:
            # s_pos is after last geometry because of rounding error
            if np.isclose(s, self.geometry_lengths[-1], 0.01, 0.01):  # todo parameter
                geo_idx = self.geometry_lengths.size - 2
            else:
                raise Exception(
                    f"Tried to calculate a position outside of the borders of the reference path at s={s}"
                    f", but path has only length of l={self.length}"
                )

        return self.geometries[geo_idx].interpolate_se2(s - self.geometry_lengths[geo_idx], t)


@dataclass
class Border:

    reference: Union[Border, PlanView]
    width_coefficient_offsets: List[float]
    width_coefficients: List[List[float]]

    s_offset: float = 0.0

    # NOTE: loaded in __post_init__
    length: Optional[float] = None

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        self.length = float(self.reference.length)

    @classmethod
    def from_plan_view(cls, plan_view: PlanView, lane_offsets: List[LaneOffset]) -> Border:
        args = {}
        args["reference"] = plan_view

        width_coefficient_offsets = []
        width_coefficients = []

        # Lane offsets will be coeffs
        # this has to be done if the reference path has the laneoffset attribute
        # and thus is different to the geometry described in the plan_view
        # openDRIVE lets multiple laneOffsets start at the same position
        # but only the last one counts -> delete all previous ones
        if any(lane_offsets):
            for lane_offset in lane_offsets:
                if lane_offset.s in width_coefficient_offsets:
                    # offset is already there, delete previous entries
                    idx = width_coefficient_offsets.index(lane_offset.s)
                    del width_coefficient_offsets[idx]
                    del width_coefficients[idx]
                width_coefficient_offsets.append(lane_offset.s)
                width_coefficients.append(lane_offset.polynomial_coefficients)
        else:
            width_coefficient_offsets.append(0.0)
            width_coefficients.append([0.0])

        args["width_coefficient_offsets"] = width_coefficient_offsets
        args["width_coefficients"] = width_coefficients

        return Border(**args)

    def _get_width_index(self, s: float, is_last_pos: bool) -> float:
        """Get the index of the width which applies at position s_pos.

        :param s_pos: Position on border in curve_parameter ds
        :param is_last_pos: Whether s_pos is the last position
        :return: Index of width that applies at position s_pos
        """

        return next(
            (
                self.width_coefficient_offsets.index(n)
                for n in self.width_coefficient_offsets[::-1]
                if ((n <= s and (not is_last_pos or s == 0)) or (n < s and is_last_pos))
            ),
            len(self.width_coefficient_offsets) - 1,
        )

    def interpolate_se2(self, s: float, t: float = 0.0, is_last_pos: bool = False) -> npt.NDArray[np.float64]:
        # Last reference has to be a reference geometry (PlanView)
        # Offset of all inner lanes (Border)
        # calculate position of reference border
        if np.isclose(s, 0):
            s = 0

        try:
            se2 = self.reference.interpolate_se2(self.s_offset + s, is_last_pos=is_last_pos)
        except TypeError:
            se2 = self.reference.interpolate_se2(np.round(self.s_offset + s, 3), is_last_pos=is_last_pos)

        if len(self.width_coefficients) == 0 or len(self.width_coefficient_offsets) == 0:
            raise Exception("No entries for width definitions.")

        # Find correct coefficients
        # find which width segment is at s_pos
        width_idx = self._get_width_index(s, is_last_pos=is_last_pos)
        # width_idx = min(width_idx, len(self.width_coefficient_offsets)-1)
        # Calculate width at s_pos
        distance = (
            np.polynomial.polynomial.polyval(
                s - self.width_coefficient_offsets[width_idx],
                self.width_coefficients[width_idx],
            )
            + t
        )
        ortho = normalize_angle(se2[StateSE2Index.YAW] + np.pi / 2)
        se2[StateSE2Index.X] += distance * np.cos(ortho)
        se2[StateSE2Index.Y] += distance * np.sin(ortho)

        se2[StateSE2Index.YAW] = normalize_angle(se2[StateSE2Index.YAW])
        return se2
