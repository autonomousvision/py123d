"""TruckDrive map parser."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from py123d.datatypes import BaseMapObject, Lane, MapMetadata, RoadLine
from py123d.datatypes.map_objects.map_layer_types import LaneType, RoadLineType
from py123d.geometry import Polyline3D
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.truckdrive.truckdrive_constants import DATASET_NAME, VEHICLE_FRAME, VELODYNE_FRAME
from py123d.parser.truckdrive.truckdrive_log_parser import resolve_truckdrive_split
from py123d.parser.truckdrive.utils.truckdrive_helper import (
    TransformTree,
    load_gt_trajectory,
    parse_synced_filename,
    transform_points,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregatedLaneLine:
    """Aggregated lane-line geometry in global coordinates."""

    obj_id: int
    points_global: np.ndarray
    lane_line_type: str = "implicit_lane_line"
    degraded: bool = False


@dataclass
class AggregatedLaneSegment:
    """Aggregated lane-segment geometry and topology in global coordinates."""

    obj_id: int
    centerline_global: np.ndarray
    left_line: Optional[int] = None
    right_line: Optional[int] = None
    prev_lanes: List[int] = field(default_factory=list)
    next_lanes: List[int] = field(default_factory=list)
    left_lanes: List[int] = field(default_factory=list)
    right_lanes: List[int] = field(default_factory=list)
    current_lane: bool = False
    has_stop_line: bool = False


def _parse_ref_ids(value: Union[int, str, None]) -> List[int]:
    if value is None or value == "":
        return []
    if isinstance(value, int):
        return [value]
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


class TruckDriveMapParser(BaseMapParser):
    """Lightweight handle to one TruckDrive scene map."""

    def __init__(self, data_root: Path, scene_name: str, split: Optional[str] = None) -> None:
        self._data_root = Path(data_root)
        self._scene_name = scene_name
        self._scene_dir = self._data_root / scene_name
        self._split = split or resolve_truckdrive_split(scene_name)
        self._lane_lines, self._lane_segments = self._aggregate_lane_annotations()

    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return MapMetadata(
            dataset=DATASET_NAME,
            split=self._split,
            log_name=self._scene_name,
            location=self._scene_name,
            map_has_z=True,
            map_is_per_log=True,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        for lane_line in self._lane_lines.values():
            if lane_line.points_global.shape[0] < 2:
                continue
            yield RoadLine(
                object_id=lane_line.obj_id,
                road_line_type=RoadLineType.UNKNOWN,
                polyline=Polyline3D.from_array(lane_line.points_global.astype(np.float64)),
            )

        for lane_segment in self._lane_segments.values():
            if lane_segment.centerline_global.shape[0] < 2:
                continue

            centerline = Polyline3D.from_array(lane_segment.centerline_global.astype(np.float64))
            left_boundary = self._boundary_for_lane_segment(lane_segment.left_line, centerline)
            right_boundary = self._boundary_for_lane_segment(lane_segment.right_line, centerline)

            yield Lane(
                object_id=lane_segment.obj_id,
                lane_type=LaneType.SURFACE_STREET,
                lane_group_id=lane_segment.obj_id,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                centerline=centerline,
                left_lane_id=lane_segment.left_lanes[0] if lane_segment.left_lanes else None,
                right_lane_id=lane_segment.right_lanes[0] if lane_segment.right_lanes else None,
                predecessor_ids=lane_segment.prev_lanes,
                successor_ids=lane_segment.next_lanes,
                speed_limit_mps=None,
            )

    def _boundary_for_lane_segment(
        self,
        lane_line_id: Optional[int],
        fallback_centerline: Polyline3D,
    ) -> Polyline3D:
        if lane_line_id is not None and lane_line_id in self._lane_lines:
            points = self._lane_lines[lane_line_id].points_global
            if points.shape[0] >= 2:
                return Polyline3D.from_array(points.astype(np.float64))
        return fallback_centerline

    def _aggregate_lane_annotations(
        self,
    ) -> tuple[Dict[int, AggregatedLaneLine], Dict[int, AggregatedLaneSegment]]:
        lane_lines: Dict[int, AggregatedLaneLine] = {}
        lane_segments: Dict[int, AggregatedLaneSegment] = {}

        lane_dir = self._scene_dir / "annotations" / "lane_lines"
        if not lane_dir.is_dir():
            return lane_lines, lane_segments

        transform_tree = TransformTree(self._scene_dir / "calibrations" / "calib_tf_tree_full.json")
        trajectory = load_gt_trajectory(self._scene_dir / "poses" / "gt_trajectory.txt")
        velodyne_to_vehicle = transform_tree.lookup(VELODYNE_FRAME, VEHICLE_FRAME)

        for json_path in sorted(lane_dir.glob("*.json")):
            sync_id, _timestamp_ns = parse_synced_filename(json_path.name)
            if sync_id is None:
                continue
            trajectory_pose = trajectory.get(sync_id)
            if trajectory_pose is None:
                continue

            lidar_aeva_to_vehicle = transform_tree.lookup(
                "lidar_aeva_forward_center_wide",
                VEHICLE_FRAME,
            )
            vehicle_to_global = rel_to_abs_se3(
                origin=trajectory_pose.pose,
                pose_se3=lidar_aeva_to_vehicle.inverse,
            )
            velodyne_to_global = rel_to_abs_se3(
                origin=vehicle_to_global,
                pose_se3=velodyne_to_vehicle,
            ).transformation_matrix

            with json_path.open("r", encoding="utf-8") as file:
                raw_objects = json.load(file)
            objects = list(raw_objects.values()) if isinstance(raw_objects, dict) else raw_objects

            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                obj_class = obj.get("obj_class")
                obj_id = int(obj.get("obj_id"))
                points = np.asarray(obj.get("points", []), dtype=np.float64)
                if points.size == 0:
                    continue
                points_global = transform_points(velodyne_to_global, points.reshape(-1, 3))

                if obj_class == "lane_line":
                    existing = lane_lines.get(obj_id)
                    if existing is None or points_global.shape[0] > existing.points_global.shape[0]:
                        lane_lines[obj_id] = AggregatedLaneLine(
                            obj_id=obj_id,
                            points_global=points_global,
                            lane_line_type=str(obj.get("lane_line_type", "implicit_lane_line")),
                            degraded=bool(obj.get("degraded", False)),
                        )
                elif obj_class == "lane_segment":
                    left_refs = _parse_ref_ids(obj.get("left_line"))
                    right_refs = _parse_ref_ids(obj.get("right_line"))
                    existing = lane_segments.get(obj_id)
                    if existing is None or points_global.shape[0] > existing.centerline_global.shape[0]:
                        lane_segments[obj_id] = AggregatedLaneSegment(
                            obj_id=obj_id,
                            centerline_global=points_global,
                            left_line=left_refs[0] if left_refs else None,
                            right_line=right_refs[0] if right_refs else None,
                            prev_lanes=_parse_ref_ids(obj.get("prev_lanes")),
                            next_lanes=_parse_ref_ids(obj.get("next_lanes")),
                            left_lanes=_parse_ref_ids(obj.get("left_lanes")),
                            right_lanes=_parse_ref_ids(obj.get("right_lanes")),
                            current_lane=bool(obj.get("current_lane", False)),
                            has_stop_line=bool(obj.get("has_stop_line", False)),
                        )

        return lane_lines, lane_segments
