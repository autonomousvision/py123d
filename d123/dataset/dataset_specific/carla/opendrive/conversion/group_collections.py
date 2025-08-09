from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from d123.common.geometry.base import StateSE2Index
from d123.common.geometry.units import kmph_to_mps, mph_to_mps
from d123.dataset.dataset_specific.carla.opendrive.conversion.id_system import (
    derive_lane_group_id,
    derive_lane_id,
    lane_group_id_from_lane_id,
)
from d123.dataset.dataset_specific.carla.opendrive.elements.lane import Lane, LaneSection
from d123.dataset.dataset_specific.carla.opendrive.elements.reference import Border
from d123.dataset.dataset_specific.carla.opendrive.elements.road import RoadType

# TODO: Add to config
STEP_SIZE = 1.0

# TODO: make naming consistent with objects_collections.py


@dataclass
class OpenDriveLaneHelper:

    lane_id: str
    open_drive_lane: Lane
    s_inner_offset: float
    s_range: Tuple[float, float]
    inner_border: Border
    outer_border: Border
    speed_limit_mps: Optional[float]
    reverse: bool = False

    # lazy loaded
    predecessor_lane_ids: Optional[List[str]] = None
    successor_lane_ids: Optional[List[str]] = None

    def __post_init__(self):
        self.predecessor_lane_ids: List[str] = []
        self.successor_lane_ids: List[str] = []

    @property
    def id(self) -> int:
        return self.open_drive_lane.id

    @property
    def type(self) -> str:
        return self.open_drive_lane.type

    @cached_property
    def _s_positions(self) -> npt.NDArray[np.float64]:
        length = self.s_range[1] - self.s_range[0]
        _s_positions = np.linspace(
            self.s_range[0], self.s_range[1], int(np.ceil(length / STEP_SIZE)) + 1, endpoint=True, dtype=np.float64
        )
        _s_positions[..., -1] = np.clip(_s_positions[..., -1], 0.0, self.s_range[-1])
        return _s_positions

    @cached_property
    def _is_last_mask(self) -> npt.NDArray[np.float64]:
        is_last_mask = np.zeros(len(self._s_positions), dtype=bool)
        is_last_mask[-1] = True
        return is_last_mask

    @cached_property
    def inner_polyline_se2(self) -> npt.NDArray[np.float64]:
        inner_polyline = np.array(
            [
                self.inner_border.interpolate_se2(self.s_inner_offset + s - self.s_range[0], is_last_pos=is_last)
                for s, is_last in zip(self._s_positions, self._is_last_mask)
            ],
            dtype=np.float64,
        )
        return np.flip(inner_polyline, axis=0) if self.id > 0 else inner_polyline

    @cached_property
    def inner_polyline_3d(self) -> npt.NDArray[np.float64]:
        inner_polyline = np.array(
            [
                self.inner_border.interpolate_3d(self.s_inner_offset + s - self.s_range[0], is_last_pos=is_last)
                for s, is_last in zip(self._s_positions, self._is_last_mask)
            ],
            dtype=np.float64,
        )
        return np.flip(inner_polyline, axis=0) if self.id > 0 else inner_polyline

    @cached_property
    def outer_polyline_se2(self) -> npt.NDArray[np.float64]:
        outer_polyline = np.array(
            [
                self.outer_border.interpolate_se2(s - self.s_range[0], is_last_pos=is_last)
                for s, is_last in zip(self._s_positions, self._is_last_mask)
            ],
            dtype=np.float64,
        )
        return np.flip(outer_polyline, axis=0) if self.id > 0 else outer_polyline

    @cached_property
    def outer_polyline_3d(self) -> npt.NDArray[np.float64]:
        outer_polyline = np.array(
            [
                self.outer_border.interpolate_3d(s - self.s_range[0], is_last_pos=is_last)
                for s, is_last in zip(self._s_positions, self._is_last_mask)
            ],
            dtype=np.float64,
        )
        return np.flip(outer_polyline, axis=0) if self.id > 0 else outer_polyline

    @property
    def center_polyline_se2(self) -> npt.NDArray[np.float64]:
        return np.concatenate([self.inner_polyline_se2[None, ...], self.outer_polyline_se2[None, ...]], axis=0).mean(
            axis=0
        )

    @property
    def center_polyline_3d(self) -> npt.NDArray[np.float64]:
        return np.concatenate([self.outer_polyline_3d[None, ...], self.inner_polyline_3d[None, ...]], axis=0).mean(
            axis=0
        )

    @property
    def outline_polyline_3d(self) -> npt.NDArray[np.float64]:
        inner_polyline = self.inner_polyline_3d[::-1]
        outer_polyline = self.outer_polyline_3d
        return np.concatenate([inner_polyline, outer_polyline, inner_polyline[None, 0]], axis=0, dtype=np.float64)

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        inner_polyline = self.inner_polyline_se2[..., StateSE2Index.XY][::-1]
        outer_polyline = self.outer_polyline_se2[..., StateSE2Index.XY]
        polygon_exterior = np.concatenate([inner_polyline, outer_polyline], axis=0, dtype=np.float64)

        return shapely.Polygon(polygon_exterior)


@dataclass
class OpenDriveLaneGroupHelper:

    lane_group_id: str
    lane_helpers: List[OpenDriveLaneHelper]

    # loaded during __post_init__
    predecessor_lane_group_ids: Optional[List[str]] = None
    successor_lane_group_ids: Optional[List[str]] = None
    junction_id: Optional[int] = None

    def __post_init__(self):

        predecessor_lane_group_ids = []
        successor_lane_group__ids = []
        for lane_helper in self.lane_helpers:
            for predecessor_lane_id in lane_helper.predecessor_lane_ids:
                predecessor_lane_group_ids.append(lane_group_id_from_lane_id(predecessor_lane_id))
            for successor_lane_id in lane_helper.successor_lane_ids:
                successor_lane_group__ids.append(lane_group_id_from_lane_id(successor_lane_id))
        self.predecessor_lane_group_ids: List[str] = list(set(predecessor_lane_group_ids))
        self.successor_lane_group_ids: List[str] = list(set(successor_lane_group__ids))

        assert len(set([lane_group_id_from_lane_id(lane_helper.lane_id) for lane_helper in self.lane_helpers])) == 1

    def _get_inner_lane_helper(self) -> OpenDriveLaneHelper:
        lane_helper_ids = [lane_helper.open_drive_lane.id for lane_helper in self.lane_helpers]
        inner_lane_helper_idx = np.argmin(lane_helper_ids) if lane_helper_ids[0] > 0 else np.argmax(lane_helper_ids)
        return self.lane_helpers[inner_lane_helper_idx]

    def _get_outer_lane_helper(self) -> OpenDriveLaneHelper:
        lane_helper_ids = [lane_helper.open_drive_lane.id for lane_helper in self.lane_helpers]
        outer_lane_helper_idx = np.argmax(lane_helper_ids) if lane_helper_ids[0] > 0 else np.argmin(lane_helper_ids)
        return self.lane_helpers[outer_lane_helper_idx]

    @cached_property
    def inner_polyline_se2(self):
        return self._get_inner_lane_helper().inner_polyline_se2

    @cached_property
    def outer_polyline_se2(self):
        return self._get_outer_lane_helper().outer_polyline_se2

    @cached_property
    def inner_polyline_3d(self):
        return self._get_inner_lane_helper().inner_polyline_3d

    @cached_property
    def outer_polyline_3d(self):
        return self._get_outer_lane_helper().outer_polyline_3d

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        inner_polyline = self.inner_polyline_se2[..., StateSE2Index.XY][::-1]
        outer_polyline = self.outer_polyline_se2[..., StateSE2Index.XY]
        polygon_exterior = np.concatenate([inner_polyline, outer_polyline], axis=0, dtype=np.float64)
        return shapely.Polygon(polygon_exterior)


def lane_section_to_lane_helpers(
    lane_section_id: str,
    lane_section: LaneSection,
    reference_border: Border,
    s_min: float,
    s_max: float,
    road_types: List[RoadType],
) -> Dict[str, OpenDriveLaneHelper]:

    lane_helpers: Dict[str, OpenDriveLaneHelper] = {}

    for side in ["right", "left"]:
        lane_group_id = derive_lane_group_id(lane_section_id, side)
        lanes = lane_section.right_lanes if side == "right" else lane_section.left_lanes
        coeff_factor = -1.0 if side == "right" else 1.0

        lane_borders = [reference_border]

        for lane in lanes:
            lane_id = derive_lane_id(lane_group_id, lane.id)
            s_inner_offset = lane_section.s if len(lane_borders) == 1 else 0.0
            lane_borders.append(_create_outer_lane_border(lane_borders, lane_section, lane, coeff_factor))
            lane_helper = OpenDriveLaneHelper(
                lane_id=lane_id,
                open_drive_lane=lane,
                s_inner_offset=s_inner_offset,
                s_range=(s_min, s_max),
                inner_border=lane_borders[-2],
                outer_border=lane_borders[-1],
                speed_limit_mps=_get_speed_limit_mps(s_min, s_max, road_types),
            )
            lane_helpers[lane_id] = lane_helper

    return lane_helpers


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
    args["elevation_profile"] = lane_borders[-1].elevation_profile

    width_coefficient_offsets = []
    width_coefficients = []

    for width in lane.widths:
        width_coefficient_offsets.append(width.s_offset)
        width_coefficients.append([x * coeff_factor for x in width.polynomial_coefficients])

    args["width_coefficient_offsets"] = width_coefficient_offsets
    args["width_coefficients"] = width_coefficients
    return Border(**args)


def _get_speed_limit_mps(s_min: float, s_max: float, road_types: List[RoadType]) -> Optional[float]:

    # NOTE: Likely not correct way to extract speed limit from CARLA maps, but serves as a placeholder
    speed_limit_mps: Optional[float] = None
    s_road_types = [road_type.s for road_type in road_types] + [float("inf")]

    if len(road_types) > 0:
        # 1. Find current road type
        for road_type_idx, road_type in enumerate(road_types):
            if s_min >= road_type.s and s_min < s_road_types[road_type_idx + 1]:
                if road_type.speed is not None:
                    if road_type.speed.unit == "mps":
                        speed_limit_mps = road_type.speed.max
                    elif road_type.speed.unit == "km/h":
                        speed_limit_mps = kmph_to_mps(road_type.speed.max)
                    elif road_type.speed.unit == "mph":
                        speed_limit_mps = mph_to_mps(road_type.speed.max)
                break
    return speed_limit_mps
