import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from py123d.conversion.utils.map_utils.opendrive.parser.reference import XODRReferenceLine
from py123d.conversion.utils.map_utils.opendrive.parser.road import XODRRoad
from py123d.conversion.utils.map_utils.opendrive.parser.signals import XODRSignal, XODRSignalReference
from py123d.conversion.utils.map_utils.opendrive.utils.id_system import build_lane_id
from py123d.geometry import Point3D
from py123d.geometry.utils.rotation_utils import normalize_angle

logger = logging.getLogger(__name__)


@dataclass
class OpenDriveSignalHelper:
    signal_id: int
    signal_type: str
    lane_ids: List[str]  # Lane IDs controlled by this signal
    turn_relation: Optional[str]
    position_3d: npt.NDArray[np.float64]  # Signal reference position [x, y, z]
    heading: float
    reference_s: float  # s coordinate of signal reference (for stop zone placement)
    xodr_signal: XODRSignal


def _lane_section_idx_from_s(road: XODRRoad, s: float) -> int:
    lane_section_idx = 0
    for idx, lane_section in enumerate(road.lanes.lane_sections):
        if s < lane_section.s:
            break
        lane_section_idx = idx
    return lane_section_idx


def _lane_ids_from_signal_ref_validity(
    signal_ref: XODRSignalReference,
    road: XODRRoad,
) -> List[str]:
    """Extract lane IDs from signal reference validity elements."""
    lane_section_idx = _lane_section_idx_from_s(road, signal_ref.s)

    if not signal_ref.validity:
        return []

    lane_indices = set()
    for validity in signal_ref.validity:
        if validity.from_lane == validity.to_lane:
            lane_indices.add(validity.from_lane)
            continue
        step = 1 if validity.to_lane > validity.from_lane else -1
        lane_indices.update(range(validity.from_lane, validity.to_lane + step, step))

    # Remove center lane (not drivable)
    lane_indices.discard(0)

    if not lane_indices:
        return []

    return [build_lane_id(road.id, lane_section_idx, lane_idx) for lane_idx in sorted(lane_indices)]


def get_signal_reference_helper(
    signal_ref: XODRSignalReference,
    signal_lookup: Dict[int, XODRSignal],
    reference_line: XODRReferenceLine,
    road: XODRRoad,
) -> Optional[OpenDriveSignalHelper]:
    """Create helper from signal reference (has lane validity) and signal definition (has type)."""
    # Get signal definition for type info
    signal = signal_lookup.get(signal_ref.id)
    if signal is None:
        logger.debug(f"Signal definition not found for signal_ref.id={signal_ref.id} on road {road.id}")
        return None

    signal_s = float(np.clip(signal_ref.s, 0.0, reference_line.length))

    # Use signal reference's s, t for position (where stop zone goes)
    se2 = reference_line.interpolate_se2(s=signal_s, t=signal_ref.t)
    point_3d = Point3D.from_array(reference_line.interpolate_3d(s=signal_s, t=signal_ref.t))
    position_3d = np.array([se2[0], se2[1], point_3d.z], dtype=np.float64)

    # Compute heading from reference line yaw
    heading = se2[2]
    if signal_ref.orientation == "-":
        heading = normalize_angle(heading + np.pi)

    # Get lane IDs from signal reference validity
    lane_ids = _lane_ids_from_signal_ref_validity(signal_ref, road)

    return OpenDriveSignalHelper(
        signal_id=signal_ref.id,
        signal_type=signal.type,
        lane_ids=lane_ids,
        turn_relation=signal_ref.turn_relation,
        position_3d=position_3d,
        heading=heading,
        reference_s=signal_s,
        xodr_signal=signal,
    )
