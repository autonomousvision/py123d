import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from shapely.ops import polygonize, unary_union

from d123.common.geometry.base import Point3DIndex
from d123.common.geometry.occupancy_map import OccupancyMap2D
from d123.dataset.dataset_specific.carla.opendrive.conversion.group_collections import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
    lane_section_to_lane_helpers,
)
from d123.dataset.dataset_specific.carla.opendrive.conversion.id_system import (
    build_lane_id,
    derive_lane_section_id,
    lane_group_id_from_lane_id,
    road_id_from_lane_group_id,
)
from d123.dataset.dataset_specific.carla.opendrive.conversion.objects_collections import (
    OpenDriveObjectHelper,
    get_object_helper,
)
from d123.dataset.dataset_specific.carla.opendrive.elements.opendrive import Junction, OpenDrive, Road
from d123.dataset.dataset_specific.carla.opendrive.elements.reference import Border
from d123.dataset.dataset_specific.carla.opendrive.id_mapping import IntIDMapping
from d123.dataset.dataset_specific.nuplan.nuplan_map_conversion import get_road_edge_linestrings
from d123.dataset.maps.map_datatypes import MapSurfaceType

ENABLE_WARNING: bool = False
CONNECTION_DISTANCE_THRESHOLD: float = 0.1  # [m]
D123_MAPS_ROOT = Path(os.environ.get("D123_MAPS_ROOT"))


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

        road_edge_df = self._extract_road_edge_df(lane_df, carpark_df, generic_drivable_area_df, lane_group_df)

        # Store dataframes
        map_file_name = D123_MAPS_ROOT / f"{map_name}.gpkg"
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
        road_edge_df.to_file(map_file_name, layer=MapSurfaceType.ROAD_EDGE.serialize(), driver="GPKG", mode="a")

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

                # assert successor_lane_id in self.lane_helper_dict.keys()
                if successor_lane_id is None or successor_lane_id not in self.lane_helper_dict.keys():
                    continue
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

                # assert predecessor_lane_id in self.lane_helper_dict.keys()
                if predecessor_lane_id is None or predecessor_lane_id not in self.lane_helper_dict.keys():
                    continue
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

                    if incoming_lane_id is None or connecting_lane_id is None:
                        continue
                    if (
                        incoming_lane_id not in self.lane_helper_dict.keys()
                        or connecting_lane_id not in self.lane_helper_dict.keys()
                    ):
                        if ENABLE_WARNING:
                            warnings.warn(
                                f"Warning..... Lane connection {incoming_lane_id} -> {connecting_lane_id} not found in lane_helper_dict"
                            )
                        continue
                    # assert incoming_lane_id in self.lane_helper_dict.keys()
                    # assert connecting_lane_id in self.lane_helper_dict.keys()
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

        def _collect_lane_group_ids_of_road(road_id: int) -> List[str]:
            lane_group_ids: List[str] = []
            for lane_group_id in self.lane_group_helper_dict.keys():
                if int(road_id_from_lane_group_id(lane_group_id)) == road_id:
                    lane_group_ids.append(lane_group_id)
            return lane_group_ids

        for junction in self.junction_dict.values():
            for connection in junction.connections:
                connecting_road = self.road_dict[connection.connecting_road]
                connecting_lane_group_ids = _collect_lane_group_ids_of_road(connecting_road.id)
                for connecting_lane_group_id in connecting_lane_group_ids:
                    if connecting_lane_group_id in self.lane_group_helper_dict.keys():
                        self.lane_group_helper_dict[connecting_lane_group_id].junction_id = junction.id

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
        outlines = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "sidewalk":
                ids.append(lane_helper.lane_id)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                outlines.append(shapely.LineString(lane_helper.outline_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                # "left_boundary": left_boundaries,
                # "right_boundary": left_boundaries,
                "outline": outlines,
            }
        )
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_carpark_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        left_boundaries = []
        right_boundaries = []
        outlines = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type == "parking":
                ids.append(lane_helper.lane_id)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                outlines.append(shapely.LineString(lane_helper.outline_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
                "outline": outlines,
            }
        )
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_generic_drivable_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        left_boundaries = []
        right_boundaries = []
        outlines = []
        geometries = []

        for lane_helper in self.lane_helper_dict.values():
            if lane_helper.type in ["none", "border", "bidirectional"]:
                ids.append(lane_helper.lane_id)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                outlines.append(shapely.LineString(lane_helper.outline_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
                "outline": outlines,
            }
        )
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_intersections_dataframe(self) -> gpd.GeoDataFrame:
        def _find_lane_group_helpers_with_junction_id(junction_id: int) -> List[OpenDriveLaneGroupHelper]:
            return [
                lane_group_helper
                for lane_group_helper in self.lane_group_helper_dict.values()
                if lane_group_helper.junction_id == junction_id
            ]

        ids = []
        lane_group_ids = []
        geometries = []
        for junction in self.junction_dict.values():
            lane_group_helpers = _find_lane_group_helpers_with_junction_id(junction.id)
            lane_group_ids_ = [lane_group_helper.lane_group_id for lane_group_helper in lane_group_helpers]
            if len(lane_group_ids_) == 0:
                warnings.warn(f"Skipped Junction {junction.id} without drivable lanes!")
                continue

            polygon = extract_exteriors_polygon(lane_group_helpers)
            ids.append(junction.id)
            lane_group_ids.append(lane_group_ids_)
            geometries.append(polygon)

        data = pd.DataFrame({"id": ids, "lane_group_ids": lane_group_ids})
        return gpd.GeoDataFrame(data, geometry=geometries)

    def _extract_lane_group_dataframe(self) -> gpd.GeoDataFrame:

        ids = []
        lane_ids = []
        predecessor_lane_group_ids = []
        successor_lane_group_ids = []
        intersection_ids = []
        left_boundaries = []
        right_boundaries = []
        geometries = []

        for lane_group_helper in self.lane_group_helper_dict.values():
            lane_group_helper: OpenDriveLaneGroupHelper
            ids.append(lane_group_helper.lane_group_id)
            lane_ids.append([lane_helper.lane_id for lane_helper in lane_group_helper.lane_helpers])
            predecessor_lane_group_ids.append(lane_group_helper.predecessor_lane_group_ids)
            successor_lane_group_ids.append(lane_group_helper.successor_lane_group_ids)
            intersection_ids.append(lane_group_helper.junction_id)
            left_boundaries.append(shapely.LineString(lane_group_helper.inner_polyline_3d))
            right_boundaries.append(shapely.LineString(lane_group_helper.outer_polyline_3d))
            geometries.append(lane_group_helper.shapely_polygon)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_ids": lane_ids,
                "predecessor_ids": predecessor_lane_group_ids,
                "successor_ids": successor_lane_group_ids,
                "intersection_id": intersection_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
            }
        )
        gdf = gpd.GeoDataFrame(data, geometry=geometries)

        return gdf

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

        # NOTE: intersection and crosswalk ids are already integers

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

        intersections_df["lane_group_ids"] = intersections_df["lane_group_ids"].apply(
            lambda x: lane_group_id_mapping.map_list(x)
        )

    def _extract_road_edge_df(
        self,
        lane_df: gpd.GeoDataFrame,
        carpark_df: gpd.GeoDataFrame,
        generic_drivable_area_df: gpd.GeoDataFrame,
        lane_group_df: gpd.GeoDataFrame,
    ) -> None:
        road_edges = _get_road_edges_from_gdf(lane_df, carpark_df, generic_drivable_area_df, lane_group_df)

        ids = np.arange(len(road_edges), dtype=np.int64).tolist()
        geometries = road_edges
        return gpd.GeoDataFrame(pd.DataFrame({"id": ids}), geometry=geometries)


# TODO: move this somewhere else and improve
def extract_exteriors_polygon(lane_group_helpers: List[OpenDriveLaneGroupHelper]) -> shapely.Polygon:

    # Step 1: Extract all boundary line segments
    all_polygons = []
    for lane_group_helper in lane_group_helpers:
        all_polygons.append(lane_group_helper.shapely_polygon)

    # Step 2: Merge all boundaries and extract the enclosed polygons
    # try:
    merged_boundaries = unary_union(all_polygons)
    # except Exception as e:
    #     warnings.warn(f"Topological error during polygon union: {e}")
    #     print([(helper.lane_group_id, poly.is_valid) for poly, helper in zip(all_polygons, lane_group_helpers)])
    #     merged_boundaries = unary_union([poly for poly in all_polygons if poly.is_valid])

    # Step 3: Generate polygons from the merged lines
    polygons = list(polygonize(merged_boundaries))

    # Step 4: Select the polygon that represents the intersection
    # Usually it's the largest polygon
    if len(polygons) == 1:
        return polygons[0]
    else:
        # Take the largest polygon if there are multiple
        return max(polygons, key=lambda p: p.area)


def _get_road_edges_from_gdf(
    lane_df: gpd.GeoDataFrame,
    carpark_df: gpd.GeoDataFrame,
    generic_drivable_area_df: gpd.GeoDataFrame,
    lane_group_df: gpd.GeoDataFrame,
) -> List[shapely.LineString]:

    # 1. Find conflicting lane groups, e.g. groups of lanes that overlap in 2D but have different Z-values (bridges)
    conflicting_lane_groups = _get_conflicting_lane_groups(lane_group_df, lane_df)

    # 2. Extract road edges in 2D (including conflicting lane groups)
    drivable_polygons = (
        lane_group_df.geometry.tolist() + carpark_df.geometry.tolist() + generic_drivable_area_df.geometry.tolist()
    )
    road_edges_2d = get_road_edge_linestrings(drivable_polygons, max_road_edge_length=None)

    # 3. Collect 3D boundaries of non-conflicting lane groups and other drivable areas
    non_conflicting_boundaries: List[shapely.LineString] = []
    for lane_group_id, lane_group_helper in lane_group_df.iterrows():
        if lane_group_id not in conflicting_lane_groups.keys():
            non_conflicting_boundaries.append(lane_group_helper["left_boundary"])
            non_conflicting_boundaries.append(lane_group_helper["right_boundary"])
    for outline in carpark_df.outline.tolist() + generic_drivable_area_df.outline.tolist():
        non_conflicting_boundaries.append(outline)

    # 4. Lift road edges to 3D using the boundaries of non-conflicting elements
    non_conflicting_road_edges = lift_road_edges_to_3d(road_edges_2d, non_conflicting_boundaries)

    # 5. Add road edges from conflicting lane groups
    resolved_road_edges = _resolve_conflicting_lane_groups(conflicting_lane_groups, lane_group_df)

    all_road_edges = non_conflicting_road_edges + resolved_road_edges

    return all_road_edges


def _get_nearest_z_from_points_3d(points_3d: npt.NDArray[np.float64], query_point: npt.NDArray[np.float64]) -> float:
    assert points_3d.ndim == 2 and points_3d.shape[1] == len(
        Point3DIndex
    ), "points_3d must be a 2D array with shape (N, 3)"
    distances = np.linalg.norm(points_3d[..., Point3DIndex.XY] - query_point[..., Point3DIndex.XY], axis=1)
    closest_point = points_3d[np.argmin(distances)]
    return closest_point[2]


def _get_conflicting_lane_groups(lane_group_df: gpd.GeoDataFrame, lane_df: gpd.GeoDataFrame) -> Dict[int, List[int]]:
    Z_THRESHOLD = 5.0  # [m] Z-value threshold for conflict detection

    ids = lane_group_df.id.tolist()
    polygons = lane_group_df.geometry.tolist()

    def _get_centerline_points_3d(lane_group_id: int) -> npt.NDArray[np.float64]:
        """Helper function to get the centerline points in 3D."""
        lane_ids = lane_group_df[lane_group_df.id == lane_group_id].lane_ids.values[0]
        centerlines: List[npt.NDArray[np.float64]] = []
        for lane_id in lane_ids:
            centerline = lane_df[lane_df.id == lane_id].baseline_path.values[0]
            assert isinstance(centerline, shapely.LineString)
            centerlines.append(np.array(centerline.coords, dtype=np.float64))
        return np.concatenate(centerlines, axis=0)

    occupancy_map = OccupancyMap2D(polygons, ids)
    conflicting_lane_groups: Dict[int, List[int]] = defaultdict(list)

    for lane_group_id, lane_group_polygon in zip(ids, polygons):

        # Extract internal centerline points for Z-value check
        lane_group_centerlines = _get_centerline_points_3d(lane_group_id)

        # Query lane groups that overlap in 2D
        intersecting_lane_group_ids = occupancy_map.intersects(lane_group_polygon)
        intersecting_lane_group_ids.remove(lane_group_id)  # Remove self from the list
        for intersecting_id in intersecting_lane_group_ids:
            intersecting_geometry = occupancy_map[intersecting_id]

            # ignore non-polygon geometries
            if intersecting_geometry.geom_type != "Polygon":
                continue

            # Check if Z-values deviate at intersection centroid
            intersection_centroid = np.array(intersecting_geometry.centroid.coords[0], dtype=np.float64)
            intersecting_centerlines = _get_centerline_points_3d(intersecting_id)
            z_at_intersecting = _get_nearest_z_from_points_3d(intersecting_centerlines, intersection_centroid)
            z_at_lane_group = _get_nearest_z_from_points_3d(lane_group_centerlines, intersection_centroid)
            if np.abs(z_at_lane_group - z_at_intersecting) < Z_THRESHOLD:
                continue
            conflicting_lane_groups[lane_group_id].append(intersecting_id)
    return conflicting_lane_groups


def _resolve_conflicting_lane_groups(
    conflicting_lane_groups: Dict[int, List[int]], lane_group_df: gpd.GeoDataFrame
) -> List[shapely.LineString]:

    # Split conflicting lane groups into non-conflicting sets for further merging
    non_conflicting_sets = create_non_conflicting_sets(conflicting_lane_groups)

    road_edges_3d: List[shapely.LineString] = []
    for non_conflicting_set in non_conflicting_sets:

        # Collect 2D polygons of non-conflicting lane group set
        set_polygons = [
            lane_group_df[lane_group_df.id == lane_group_id].geometry.values[0] for lane_group_id in non_conflicting_set
        ]
        # Get 2D road edge linestrings for the non-conflicting set
        set_road_edges_2d = get_road_edge_linestrings(set_polygons, max_road_edge_length=None)

        #  Collect 3D boundaries of non-conflicting lane groups
        set_boundaries_3d: List[shapely.LineString] = []
        for lane_group_id in non_conflicting_set:
            lane_group_helper = lane_group_df[lane_group_df.id == lane_group_id]
            set_boundaries_3d.append(lane_group_helper.left_boundary.values[0])
            set_boundaries_3d.append(lane_group_helper.right_boundary.values[0])

        # Lift road edges to 3D using the boundaries of non-conflicting lane groups
        lifted_road_edges_3d = lift_road_edges_to_3d(set_road_edges_2d, set_boundaries_3d)
        road_edges_3d.extend(lifted_road_edges_3d)

    return road_edges_3d


def lift_road_edges_to_3d(
    road_edges_2d: List[shapely.LineString], boundaries: List[shapely.LineString]
) -> List[shapely.LineString]:

    QUERY_MAX_DISTANCE = 0.01  # [m] Maximum distance for nearest neighbor query

    def _find_continuous_sublists(integers: List[int]) -> List[List[int]]:
        """Find continuous sublists in a list of integers."""
        arr = np.array(integers, dtype=np.int64)
        breaks = np.where(np.diff(arr) != 1)[0] + 1
        splits = np.split(arr, breaks)
        return [sublist.tolist() for sublist in splits]

    occupancy_map = OccupancyMap2D(boundaries)

    road_edges_3d: List[shapely.LineString] = []
    for idx, linestring in enumerate(road_edges_2d):
        # print(list(linestring.coords))
        points_3d = np.array(list(linestring.coords), dtype=np.float64)

        results = occupancy_map.query_nearest(
            shapely.points(points_3d[..., :2]), max_distance=QUERY_MAX_DISTANCE, exclusive=True
        )
        for query_idx, geometry_idx in zip(*results):
            intersecting_boundary: shapely.LineString = occupancy_map[occupancy_map.ids[geometry_idx]]
            points_3d[query_idx, 2] = _get_nearest_z_from_points_3d(
                np.array(list(intersecting_boundary.coords), dtype=np.float64), points_3d[query_idx]
            )

        for continuous_slice in _find_continuous_sublists(results[0]):
            if len(continuous_slice) < 2:
                continue
            lifted_linestring = shapely.LineString(points_3d[continuous_slice])
            road_edges_3d.append(lifted_linestring)
    return road_edges_3d


def create_non_conflicting_sets(conflicts: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Creates sets of non-conflicting indices using NetworkX.
    """
    # Create graph from conflicts
    G = nx.Graph()
    for idx, conflict_list in conflicts.items():
        for conflict_idx in conflict_list:
            G.add_edge(idx, conflict_idx)

    result = []

    # Process each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)

        # Try bipartite coloring first (most common case)
        if nx.is_bipartite(subgraph):
            sets = nx.bipartite.sets(subgraph)
            result.extend([set(s) for s in sets])
        else:
            # Fall back to greedy coloring for non-bipartite graphs
            coloring = nx.greedy_color(subgraph, strategy="largest_first")
            color_groups = {}
            for node, color in coloring.items():
                if color not in color_groups:
                    color_groups[color] = set()
                color_groups[color].add(node)
            result.extend(color_groups.values())

    return result
