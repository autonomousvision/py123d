import warnings
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as geom

from asim.common.geometry.base_enum import StateSE2Index
from asim.dataset.dataset_specific.carla.opendrive.conversion.group_collections import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
    lane_section_to_lane_helpers,
)
from asim.dataset.dataset_specific.carla.opendrive.conversion.id_system import (
    build_lane_id,
    derive_lane_section_id,
    lane_group_id_from_lane_id,
)
from asim.dataset.dataset_specific.carla.opendrive.elements.opendrive import Junction, OpenDrive, Road
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import Border

ENABLE_WARNING: bool = False
CONNECTION_DISTANCE_THRESHOLD: float = 0.1  # [m]


class OpenDriveConverter:
    def __init__(self, opendrive: OpenDrive):

        self.opendrive: OpenDrive = opendrive

        self.road_dict: Dict[int, Road] = {road.id: road for road in opendrive.roads}
        self.junction_dict: Dict[int, Junction] = {junction.id: junction for junction in opendrive.junctions}

        # loaded during conversion
        self.lane_helper_dict: Dict[str, OpenDriveLaneHelper] = {}
        self.lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper] = {}

    def run(self) -> None:
        self._collect_lane_helpers()
        self._update_connection_from_links()
        self._update_connection_from_junctions()
        self._flip_and_set_connections()
        self._post_process_connections()
        self._collect_lane_groups()

    def _collect_lane_helpers(self) -> None:
        for road in self.opendrive.roads:
            reference_border = Border.from_plan_view(road.plan_view, road.lanes.lane_offsets)
            lane_section_lengths: List[float] = [ls.s for ls in road.lanes.lane_sections] + [road.length]
            for idx, lane_section in enumerate(road.lanes.lane_sections):
                lane_section_id = derive_lane_section_id(road.id, idx)
                lane_helpers_ = lane_section_to_lane_helpers(
                    lane_section_id,
                    lane_section,
                    reference_border,
                    lane_section_lengths[idx],
                    lane_section_lengths[idx + 1],
                )
                self.lane_helper_dict.update(lane_helpers_)

    def _update_connection_from_links(self) -> None:

        for lane_id in self.lane_helper_dict.keys():
            road_idx, lane_section_idx, side, lane_idx = lane_id.split("_")
            road_idx, lane_section_idx, lane_idx = int(road_idx), int(lane_section_idx), int(lane_idx)

            road = self.road_dict[road_idx]

            successor_lane_idx = self.lane_helper_dict[lane_id].open_drive_lane.successor
            if successor_lane_idx is not None:
                successor_lane_id: Optional[str] = None

                # Not the last lane section -> Next lane section in same road
                if lane_section_idx < road.lanes.last_lane_section_idx:
                    successor_lane_id = build_lane_id(
                        road_idx,
                        lane_section_idx + 1,
                        successor_lane_idx,
                    )

                # Last lane section -> Next road in first lane section
                # Try to get next road
                elif road.link.successor is not None and road.link.successor.element_type != "junction":
                    successor_road = self.road_dict[road.link.successor.element_id]

                    if road.link.successor.contact_point == "start":
                        successor_lane_section_idx = 0
                    else:
                        successor_lane_section_idx = successor_road.lanes.last_lane_section_idx

                    successor_lane_id = build_lane_id(
                        successor_road.id,
                        successor_lane_section_idx,
                        successor_lane_idx,
                    )

                assert successor_lane_id in self.lane_helper_dict.keys()
                self.lane_helper_dict[lane_id].successor_lane_ids.append(successor_lane_id)

            predecessor_lane_idx = self.lane_helper_dict[lane_id].open_drive_lane.predecessor
            if predecessor_lane_idx is not None:
                predecessor_lane_id: Optional[str] = None

                # Not the first lane section -> Previous lane section in same road
                if lane_section_idx > 0:
                    predecessor_lane_id = build_lane_id(
                        road_idx,
                        lane_section_idx - 1,
                        predecessor_lane_idx,
                    )

                # First lane section -> Previous road
                # Try to get previous road
                elif road.link.predecessor is not None and road.link.predecessor.element_type != "junction":
                    predecessor_road = self.road_dict[road.link.predecessor.element_id]

                    if road.link.predecessor.contact_point == "start":
                        predecessor_lane_section_idx = 0
                    else:
                        predecessor_lane_section_idx = predecessor_road.lanes.last_lane_section_idx

                    predecessor_lane_id = build_lane_id(
                        predecessor_road.id,
                        predecessor_lane_section_idx,
                        predecessor_lane_idx,
                    )

                assert predecessor_lane_id in self.lane_helper_dict.keys()
                self.lane_helper_dict[lane_id].predecessor_lane_ids.append(predecessor_lane_id)
                self.lane_helper_dict[predecessor_lane_id].successor_lane_ids.append(lane_id)

    def _update_connection_from_junctions(self) -> None:

        # add junctions to link_index
        # if contact_point is start, and laneId from connecting_road is negative
        # the connecting_road is the successor
        # if contact_point is start, and laneId from connecting_road is positive
        # the connecting_road is the predecessor
        # for contact_point == end it's exactly the other way
        for junction_idx, junction in self.junction_dict.items():
            for connection in junction.connections:
                incoming_road = self.road_dict[connection.incoming_road]
                connecting_road = self.road_dict[connection.connecting_road]
                contact_point = connection.contact_point
                for lane_link in connection.lane_links:

                    incoming_lane_id: Optional[str] = None
                    connecting_lane_id: Optional[str] = None

                    if contact_point == "start":
                        if lane_link.start < 0:
                            incoming_lane_section_idx = incoming_road.lanes.last_lane_section_idx
                        else:
                            incoming_lane_section_idx = 0
                        incoming_lane_id = build_lane_id(incoming_road.id, incoming_lane_section_idx, lane_link.start)
                        connecting_lane_id = build_lane_id(connecting_road.id, 0, lane_link.end)
                    else:
                        incoming_lane_id = build_lane_id(incoming_road.id, 0, lane_link.start)
                        connecting_lane_id = build_lane_id(
                            connecting_road.id, connecting_road.lanes.last_lane_section_idx, lane_link.end
                        )

                    assert incoming_lane_id in self.lane_helper_dict.keys()
                    assert connecting_lane_id in self.lane_helper_dict.keys()
                    self.lane_helper_dict[incoming_lane_id].successor_lane_ids.append(connecting_lane_id)
                    self.lane_helper_dict[connecting_lane_id].predecessor_lane_ids.append(incoming_lane_id)

    def _flip_and_set_connections(self) -> None:

        for lane_id in self.lane_helper_dict.keys():
            if self.lane_helper_dict[lane_id].id > 0:
                successors_temp = self.lane_helper_dict[lane_id].successor_lane_ids
                self.lane_helper_dict[lane_id].successor_lane_ids = self.lane_helper_dict[lane_id].predecessor_lane_ids
                self.lane_helper_dict[lane_id].predecessor_lane_ids = successors_temp
            self.lane_helper_dict[lane_id].successor_lane_ids = list(
                set(self.lane_helper_dict[lane_id].successor_lane_ids)
            )
            self.lane_helper_dict[lane_id].predecessor_lane_ids = list(
                set(self.lane_helper_dict[lane_id].predecessor_lane_ids)
            )

    def _post_process_connections(self) -> None:

        for lane_id in self.lane_helper_dict.keys():
            self.lane_helper_dict[lane_id]

            centerline = self.lane_helper_dict[lane_id].center_polyline

            valid_successor_lane_ids: List[str] = []
            for successor_lane_id in self.lane_helper_dict[lane_id].successor_lane_ids:
                successor_centerline = self.lane_helper_dict[successor_lane_id].center_polyline
                distance = np.linalg.norm(centerline[-1, :2] - successor_centerline[0, :2])
                if distance > CONNECTION_DISTANCE_THRESHOLD:
                    if ENABLE_WARNING:
                        warnings.warn(
                            f"Warning..... Removing connection {lane_id} -> {successor_lane_id} with distance {distance}"
                        )
                else:
                    valid_successor_lane_ids.append(successor_lane_id)
            self.lane_helper_dict[lane_id].successor_lane_ids = valid_successor_lane_ids

            valid_predecessor_lane_ids: List[str] = []
            for predecessor_lane_id in self.lane_helper_dict[lane_id].predecessor_lane_ids:
                predecessor_centerline = self.lane_helper_dict[predecessor_lane_id].center_polyline
                distance = np.linalg.norm(centerline[0, :2] - predecessor_centerline[-1, :2])
                if distance > CONNECTION_DISTANCE_THRESHOLD:
                    if ENABLE_WARNING:
                        warnings.warn(
                            f"Warning..... Removing connection {predecessor_lane_id} -> {successor_lane_id} with distance {distance}"
                        )
                else:
                    valid_predecessor_lane_ids.append(predecessor_lane_id)
            self.lane_helper_dict[lane_id].predecessor_lane_ids = valid_predecessor_lane_ids

    def _collect_lane_groups(self) -> None:
        def _collect_lane_helper_of_id(lane_group_id: str) -> List[OpenDriveLaneHelper]:
            lane_helpers: List[OpenDriveLaneHelper] = []
            for lane_id, lane_helper in self.lane_helper_dict.items():
                if (lane_helper.type in ["driving"]) and (lane_group_id_from_lane_id(lane_id) == lane_group_id):
                    lane_helpers.append(lane_helper)
            return lane_helpers

        all_lane_group_ids = list(
            set([lane_group_id_from_lane_id(lane_id) for lane_id in self.lane_helper_dict.keys()])
        )

        for lane_group_id in all_lane_group_ids:
            lane_group_lane_helper = _collect_lane_helper_of_id(lane_group_id)
            if len(lane_group_lane_helper) >= 1:
                self.lane_group_helper_dict[lane_group_id] = OpenDriveLaneGroupHelper(
                    lane_group_id, lane_group_lane_helper
                )

    def _extract_lane_dataframe(self) -> None:

        lane_ids = []
        predecessor_lane_ids = []
        successor_lane_ids = []
        left_boundaries = []
        right_boundaries = []
        baseline_paths = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "driving":
                lane_ids.append(lane_helper.lane_id)
                predecessor_lane_ids.append(lane_helper.predecessor_lane_ids)
                successor_lane_ids.append(lane_helper.successor_lane_ids)
                left_boundaries.append(geom.LineString(lane_helper.inner_polyline[..., StateSE2Index.XY]))
                right_boundaries.append(geom.LineString(lane_helper.outer_polyline[..., StateSE2Index.XY]))
                baseline_paths.append(geom.LineString(lane_helper.center_polyline[..., StateSE2Index.XY]))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "lane_id": lane_ids,
                "predecessor_lane_ids": predecessor_lane_ids,
                "successor_lane_ids": successor_lane_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
                "baseline_path": baseline_paths,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_walkways_dataframe(self) -> None:

        ids = []
        predecessor_ids = []
        successor_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "sidewalk":
                ids.append(lane_helper.lane_id)
                predecessor_ids.append(lane_helper.predecessor_lane_ids)
                successor_ids.append(lane_helper.successor_lane_ids)
                left_boundaries.append(geom.LineString(lane_helper.inner_polyline[..., StateSE2Index.XY]))
                right_boundaries.append(geom.LineString(lane_helper.outer_polyline[..., StateSE2Index.XY]))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_carpark_dataframe(self) -> None:
        ids = []
        predecessor_ids = []
        successor_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "parking":
                ids.append(lane_helper.lane_id)
                predecessor_ids.append(lane_helper.predecessor_lane_ids)
                successor_ids.append(lane_helper.successor_lane_ids)
                left_boundaries.append(geom.LineString(lane_helper.inner_polyline[..., StateSE2Index.XY]))
                right_boundaries.append(geom.LineString(lane_helper.outer_polyline[..., StateSE2Index.XY]))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_generic_drivable_area_dataframe(self) -> None:
        ids = []
        predecessor_ids = []
        successor_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type in ["none", "border", "bidirectional"]:
                ids.append(lane_helper.lane_id)
                predecessor_ids.append(lane_helper.predecessor_lane_ids)
                successor_ids.append(lane_helper.successor_lane_ids)
                left_boundaries.append(geom.LineString(lane_helper.inner_polyline[..., StateSE2Index.XY]))
                right_boundaries.append(geom.LineString(lane_helper.outer_polyline[..., StateSE2Index.XY]))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_intersections_dataframe(self) -> None:

        # ids = []
        # interior_lane_groups = []

        # left_boundaries = []
        # right_boundaries = []
        # geometries = []

        # for junction_idx, junction in self.junction_dict.items():
        #     for connection in junction.connections:
        #         connection.connecting_road

        #     print()
        pass

    def _extract_lane_group_dataframe(self) -> None:

        lane_group_ids = []
        predecessor_lane_group_ids = []
        successor_lane_group_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_group_helper in self.lane_group_helper_dict.values():
            lane_group_ids.append(lane_group_helper.lane_group_id)
            predecessor_lane_group_ids.append(lane_group_helper.predecessor_lane_group_ids)
            successor_lane_group_ids.append(lane_group_helper.successor_lane_group_ids)
            left_boundaries.append(geom.LineString(lane_group_helper.inner_polyline[..., StateSE2Index.XY]))
            right_boundaries.append(geom.LineString(lane_group_helper.outer_polyline[..., StateSE2Index.XY]))
            geometries.append(lane_group_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "lane_group_id": lane_group_ids,
                "predecessor_lane_group_id": predecessor_lane_group_ids,
                "successor_lane_group_id": successor_lane_group_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf
