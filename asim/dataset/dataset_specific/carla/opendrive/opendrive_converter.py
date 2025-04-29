import warnings
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

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
from asim.dataset.dataset_specific.carla.opendrive.conversion.objects_collections import (
    OpenDriveObjectHelper,
    get_object_helper,
)
from asim.dataset.dataset_specific.carla.opendrive.elements.opendrive import Junction, OpenDrive, Road
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import Border
from asim.dataset.dataset_specific.carla.opendrive.id_mapping import IntIDMapping
from asim.dataset.maps.map_datatypes import MapSurfaceType

ENABLE_WARNING: bool = False
CONNECTION_DISTANCE_THRESHOLD: float = 0.1  # [m]

# TODO:
# - add Intersections
# - add crosswalks


class OpenDriveConverter:
    def __init__(self, opendrive: OpenDrive):

        self.opendrive: OpenDrive = opendrive

        self.road_dict: Dict[int, Road] = {road.id: road for road in opendrive.roads}
        self.junction_dict: Dict[int, Junction] = {junction.id: junction for junction in opendrive.junctions}

        # loaded during conversion
        self.lane_helper_dict: Dict[str, OpenDriveLaneHelper] = {}
        self.lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper] = {}
        self.object_helper_dict: Dict[str, OpenDriveObjectHelper] = {}

    def run(self, map_name: str) -> None:

        # Run processing for map elements
        self._collect_lane_helpers()
        self._update_connection_from_links()
        self._update_connection_from_junctions()
        self._flip_and_set_connections()
        self._post_process_connections()
        self._collect_lane_groups()
        self._collect_crosswalks()

        # Collect data frames and store
        lane_df = self._extract_lane_dataframe()
        walkways_df = self._extract_walkways_dataframe()
        carpark_df = self._extract_carpark_dataframe()
        generic_drivable_area_df = self._extract_generic_drivable_dataframe()
        intersections_df = self._extract_intersections_dataframe()
        lane_group_df = self._extract_lane_group_dataframe()
        crosswalk_df = self._extract_crosswalk_dataframe()

        self._convert_ids_to_int(
            lane_df,
            walkways_df,
            carpark_df,
            generic_drivable_area_df,
            lane_group_df,
            intersections_df,
            crosswalk_df,
        )

        # Store dataframes
        map_file_name = f"{map_name}.gpkg"
        lane_df.to_file(map_file_name, layer=MapSurfaceType.LANE.serialize(), driver="GPKG")
        walkways_df.to_file(map_file_name, layer=MapSurfaceType.WALKWAY.serialize(), driver="GPKG", mode="a")
        carpark_df.to_file(map_file_name, layer=MapSurfaceType.CARPARK.serialize(), driver="GPKG", mode="a")
        generic_drivable_area_df.to_file(
            map_file_name,
            layer=MapSurfaceType.GENERIC_DRIVABLE.serialize(),
            driver="GPKG",
            mode="a",
        )
        intersections_df.to_file(map_file_name, layer=MapSurfaceType.INTERSECTION.serialize(), driver="GPKG", mode="a")
        lane_group_df.to_file(map_file_name, layer=MapSurfaceType.LANE_GROUP.serialize(), driver="GPKG", mode="a")
        crosswalk_df.to_file(map_file_name, layer=MapSurfaceType.CROSSWALK.serialize(), driver="GPKG", mode="a")

    def _collect_lane_helpers(self) -> None:
        for road in self.opendrive.roads:
            reference_border = Border.from_plan_view(road.plan_view, road.lanes.lane_offsets, road.elevation_profile)
            lane_section_lengths: List[float] = [ls.s for ls in road.lanes.lane_sections] + [road.length]
            for idx, lane_section in enumerate(road.lanes.lane_sections):
                lane_section_id = derive_lane_section_id(road.id, idx)
                lane_helpers_ = lane_section_to_lane_helpers(
                    lane_section_id,
                    lane_section,
                    reference_border,
                    lane_section_lengths[idx],
                    lane_section_lengths[idx + 1],
                    road.road_types,
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
                self.lane_helper_dict[successor_lane_id].predecessor_lane_ids.append(lane_id)

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

            centerline = self.lane_helper_dict[lane_id].center_polyline_se2

            valid_successor_lane_ids: List[str] = []
            for successor_lane_id in self.lane_helper_dict[lane_id].successor_lane_ids:
                successor_centerline = self.lane_helper_dict[successor_lane_id].center_polyline_se2
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
                predecessor_centerline = self.lane_helper_dict[predecessor_lane_id].center_polyline_se2
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

    def _collect_crosswalks(self) -> None:
        for road in self.opendrive.roads:
            if len(road.objects) == 0:
                continue
            reference_border = Border.from_plan_view(road.plan_view, road.lanes.lane_offsets, road.elevation_profile)
            for object in road.objects:
                if object.type in ["crosswalk"]:
                    object_helper = get_object_helper(object, reference_border)
                    self.object_helper_dict[object_helper.object_id] = object_helper

    def _extract_lane_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        lane_group_ids = []
        speed_limits_mps = []
        predecessor_ids = []
        successor_ids = []
        left_boundaries = []
        right_boundaries = []
        baseline_paths = []
        geometries = []

        # TODO: Extract speed limit and convert to mps
        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "driving":
                ids.append(lane_helper.lane_id)
                lane_group_ids.append(lane_group_id_from_lane_id(lane_helper.lane_id))
                speed_limits_mps.append(lane_helper.speed_limit_mps)
                predecessor_ids.append(lane_helper.predecessor_lane_ids)
                successor_ids.append(lane_helper.successor_lane_ids)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                baseline_paths.append(shapely.LineString(lane_helper.center_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_group_id": lane_group_ids,
                "speed_limit_mps": speed_limits_mps,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
                "baseline_path": baseline_paths,
            }
        )
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_walkways_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "sidewalk":
                ids.append(lane_helper.lane_id)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
            }
        )
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_carpark_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "parking":
                ids.append(lane_helper.lane_id)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
            }
        )
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_generic_drivable_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type in ["none", "border", "bidirectional"]:
                ids.append(lane_helper.lane_id)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
            }
        )
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_intersections_dataframe(self) -> gpd.GeoDataFrame:
        ids = []
        lane_group_ids = []
        geometries = []
        data = pd.DataFrame({"id": ids, "lane_group_ids": lane_group_ids})
        # TODO: Implement and extract intersection geometries
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_lane_group_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        lane_ids = []
        predecessor_lane_group_ids = []
        successor_lane_group_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_group_helper in self.lane_group_helper_dict.values():
            lane_group_helper: OpenDriveLaneGroupHelper
            ids.append(lane_group_helper.lane_group_id)
            lane_ids.append([lane_helper.lane_id for lane_helper in lane_group_helper.lane_helpers])
            predecessor_lane_group_ids.append(lane_group_helper.predecessor_lane_group_ids)
            successor_lane_group_ids.append(lane_group_helper.successor_lane_group_ids)
            left_boundaries.append(shapely.LineString(lane_group_helper.inner_polyline_3d))
            right_boundaries.append(shapely.LineString(lane_group_helper.outer_polyline_3d))
            geometries.append(lane_group_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_ids": lane_ids,
                "predecessor_ids": predecessor_lane_group_ids,
                "successor_ids": successor_lane_group_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
            }
        )

        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_crosswalk_dataframe(self) -> gpd.GeoDataFrame:
        ids = []
        outlines = []
        geometries = []
        for object_helper in self.object_helper_dict.values():
            ids.append(object_helper.object_id)
            outlines.append(shapely.LineString(object_helper.outline_3d))
            geometries.append(object_helper.shapely_polygon)

        data = pd.DataFrame({"id": ids, "outline": outlines})
        return gpd.GeoDataFrame(data, geometry=geometries)

    @staticmethod
    def _convert_ids_to_int(
        lane_df: gpd.GeoDataFrame,
        walkways_df: gpd.GeoDataFrame,
        carpark_df: gpd.GeoDataFrame,
        generic_drivable_area_df: gpd.GeoDataFrame,
        lane_group_df: gpd.GeoDataFrame,
        intersections_df: gpd.GeoDataFrame,
        crosswalk_df: gpd.GeoDataFrame,
    ) -> None:

        # initialize id mappings
        lane_id_mapping = IntIDMapping.from_series(lane_df["id"])
        walkway_id_mapping = IntIDMapping.from_series(walkways_df["id"])
        carpark_id_mapping = IntIDMapping.from_series(carpark_df["id"])
        # TODO: add id mapping for intersections
        generic_drivable_id_mapping = IntIDMapping.from_series(generic_drivable_area_df["id"])
        lane_group_id_mapping = IntIDMapping.from_series(lane_group_df["id"])

        # Adjust cross reference in lane_df and lane_group_df
        lane_df["lane_group_id"] = lane_df["lane_group_id"].map(lane_group_id_mapping.str_to_int)
        lane_group_df["lane_ids"] = lane_group_df["lane_ids"].apply(lambda x: lane_id_mapping.map_list(x))

        # Adjust predecessor/successor in lane_df and lane_group_df
        for column in ["predecessor_ids", "successor_ids"]:
            lane_df[column] = lane_df[column].apply(lambda x: lane_id_mapping.map_list(x))
            lane_group_df[column] = lane_group_df[column].apply(lambda x: lane_group_id_mapping.map_list(x))

        lane_df["id"] = lane_df["id"].map(lane_id_mapping.str_to_int)
        walkways_df["id"] = walkways_df["id"].map(walkway_id_mapping.str_to_int)
        carpark_df["id"] = carpark_df["id"].map(carpark_id_mapping.str_to_int)
        generic_drivable_area_df["id"] = generic_drivable_area_df["id"].map(generic_drivable_id_mapping.str_to_int)
        lane_group_df["id"] = lane_group_df["id"].map(lane_group_id_mapping.str_to_int)
