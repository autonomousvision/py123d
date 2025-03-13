from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from asim.dataset.dataset_specific.carla.opendrive.elements.lane import Lane, LaneSection
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import Border

step_size = 1.0


@dataclass
class ODLaneHelper:

    open_drive_lane: Lane
    s_inner_offset: float
    s_range: Tuple[float, float]
    inner_border: Border
    outer_border: Border
    reverse: bool = False

    @property
    def id(self) -> int:
        return self.open_drive_lane.id

    @property
    def type(self) -> str:
        return self.open_drive_lane.type

    @property
    def _s_positions(self) -> npt.NDArray[np.float64]:
        length = self.s_range[1] - self.s_range[0]
        return np.linspace(
            self.s_range[0],
            self.s_range[1],
            int(np.ceil(length / step_size)) + 1,
            endpoint=True,
            dtype=np.float64,
        )

    @cached_property
    def inner_polyline(self):
        s_positions = self._s_positions
        is_last_space = np.zeros(len(s_positions), dtype=bool)
        is_last_space[-1] = True

        return np.array(
            [
                self.inner_border.interpolate_se2(s - self.s_range[0], is_last_pos=is_last)
                for s, is_last in zip(s_positions, is_last_space)
            ],
            dtype=np.float64,
        )

    @cached_property
    def outer_polyline(self):
        s_positions = self._s_positions
        is_last_space = np.zeros(len(s_positions), dtype=bool)
        is_last_space[-1] = True

        return np.array(
            [
                self.outer_border.interpolate_se2(s - self.s_range[0], is_last_pos=is_last)
                for s, is_last in zip(s_positions, is_last_space)
            ],
            dtype=np.float64,
        )

    @property
    def center_polyline(self):
        return np.concatenate([self.inner_polyline[None, ...], self.outer_polyline[None, ...]], axis=0).mean(axis=0)

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        inner_polyline = self.inner_polyline[..., :2][::-1]
        outer_polyline = self.outer_polyline[..., :2]
        polygon_exterior = np.concatenate([inner_polyline, outer_polyline], axis=0, dtype=np.float64)

        return shapely.Polygon(polygon_exterior)


@dataclass
class ODLaneGroupHelper:

    id: str
    side: str
    lane_helper: List[ODLaneHelper]

    # id: str
    # type: str
    # open_drive_lane: Lane
    # inner_border: Border
    # outer_border: Border
    # reverse: bool = False


def lane_section_to_lane_group_helper(
    lane_section: LaneSection,
    reference_border: Border,
    s_min: float,
    s_max: float,
) -> List[ODLaneGroupHelper]:

    group_helpers: List[ODLaneGroupHelper] = []

    for side in ["right", "left"]:
        lane_helpers: List[ODLaneHelper] = []

        lanes = lane_section.right_lanes if side == "right" else lane_section.left_lanes
        coeff_factor = -1.0 if side == "right" else 1.0

        lane_borders = [reference_border]

        for lane in lanes:
            s_inner_offset = lane_section.s if len(lane_borders) == 1 else 0.0
            lane_borders.append(_create_outer_lane_border(lane_borders, lane_section, lane, coeff_factor))
            lane_helper = ODLaneHelper(
                open_drive_lane=lane,
                s_inner_offset=s_inner_offset,
                s_range=(s_min, s_max),
                inner_border=lane_borders[-2],
                outer_border=lane_borders[-1],
            )
            lane_helpers.append(lane_helper)
        lane_group_helper = ODLaneGroupHelper("TODO", side, lane_helpers)
        group_helpers.append(lane_group_helper)

    return group_helpers


def _create_outer_lane_border(
    lane_borders: List[Border],
    lane_section: LaneSection,
    lane: Lane,
    coeff_factor: float,
) -> Border:

    args = {}
    if len(lane_borders) == 1:
        args["s_offset"] = lane_section.s

    args["reference"] = lane_borders[-1]
    width_coefficient_offsets = []
    width_coefficients = []

    for width in lane.widths:
        width_coefficient_offsets.append(width.s_offset)
        width_coefficients.append([x * coeff_factor for x in width.polynomial_coefficients])

    args["width_coefficient_offsets"] = width_coefficient_offsets
    args["width_coefficients"] = width_coefficients
    return Border(**args)
