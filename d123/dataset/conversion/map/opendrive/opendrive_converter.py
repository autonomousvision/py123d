import os
import warnings
from pathlib import Path
from typing import Dict, Final, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import polygonize, unary_union

from d123.dataset.conversion.map.opendrive.conversion.group_collections import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
    lane_section_to_lane_helpers,
)
from d123.dataset.conversion.map.opendrive.conversion.id_system import (
    build_lane_id,
    derive_lane_section_id,
    lane_group_id_from_lane_id,
    road_id_from_lane_group_id,
)
from d123.dataset.conversion.map.opendrive.conversion.objects_collections import (
    OpenDriveObjectHelper,
    get_object_helper,
)
from d123.dataset.conversion.map.opendrive.elements.opendrive import Junction, OpenDrive, Road
from d123.dataset.conversion.map.opendrive.elements.reference import Border
from d123.dataset.conversion.map.opendrive.id_mapping import IntIDMapping
from d123.dataset.conversion.map.road_edge.road_edge_2d_utils import split_line_geometry_by_max_length
from d123.dataset.conversion.map.road_edge.road_edge_3d_utils import get_road_edges_3d_from_gdf
from d123.dataset.maps.map_datatypes import MapLayer, RoadEdgeType, RoadLineType

ENABLE_WARNING: bool = False
CONNECTION_DISTANCE_THRESHOLD: float = 0.1  # [m]
D123_MAPS_ROOT = Path(os.environ.get("D123_MAPS_ROOT"))

MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]


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
        dataframes: Dict[MapLayer, gpd.GeoDataFrame] = {}
        dataframes[MapLayer.LANE] = self._extract_lane_dataframe()
        dataframes[MapLayer.WALKWAY] = self._extract_walkways_dataframe()
        dataframes[MapLayer.CARPARK] = self._extract_carpark_dataframe()
        dataframes[MapLayer.GENERIC_DRIVABLE] = self._extract_generic_drivable_dataframe()
        dataframes[MapLayer.INTERSECTION] = self._extract_intersections_dataframe()
        dataframes[MapLayer.LANE_GROUP] = self._extract_lane_group_dataframe()
        dataframes[MapLayer.CROSSWALK] = self._extract_crosswalk_dataframe()

        self._convert_ids_to_int(
            dataframes[MapLayer.LANE],
            dataframes[MapLayer.WALKWAY],
            dataframes[MapLayer.CARPARK],
            dataframes[MapLayer.GENERIC_DRIVABLE],
            dataframes[MapLayer.LANE_GROUP],
            dataframes[MapLayer.INTERSECTION],
            dataframes[MapLayer.CROSSWALK],
        )
        dataframes[MapLayer.ROAD_EDGE] = self._extract_road_edge_df(
            dataframes[MapLayer.LANE],
            dataframes[MapLayer.CARPARK],
            dataframes[MapLayer.GENERIC_DRIVABLE],
            dataframes[MapLayer.LANE_GROUP],
        )
        dataframes[MapLayer.ROAD_LINE] = self._extract_road_line_df(
            dataframes[MapLayer.LANE],
            dataframes[MapLayer.LANE_GROUP],
        )

        # Store dataframes
        map_file_name = D123_MAPS_ROOT / f"{map_name}.gpkg"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="'crs' was not provided")
            for layer, gdf in dataframes.items():
                gdf.to_file(map_file_name, layer=layer.serialize(), driver="GPKG", mode="a")

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
        left_lane_ids = []
        right_lane_ids = []
        baseline_paths = []
        geometries = []

        for lane_group_helper in self.lane_group_helper_dict.values():
            lane_group_id = lane_group_helper.lane_group_id
            lane_helpers = lane_group_helper.lane_helpers
            num_lanes = len(lane_helpers)
            # NOTE: Lanes are going left to right
            for lane_idx, lane_helper in enumerate(lane_helpers):
                ids.append(lane_helper.lane_id)
                lane_group_ids.append(lane_group_id)
                speed_limits_mps.append(lane_helper.speed_limit_mps)
                predecessor_ids.append(lane_helper.predecessor_lane_ids)
                successor_ids.append(lane_helper.successor_lane_ids)
                left_boundaries.append(shapely.LineString(lane_helper.inner_polyline_3d))
                right_boundaries.append(shapely.LineString(lane_helper.outer_polyline_3d))
                baseline_paths.append(shapely.LineString(lane_helper.center_polyline_3d))
                geometries.append(lane_helper.shapely_polygon)
                left_lane_id = lane_helpers[lane_idx - 1].lane_id if lane_idx > 0 else None
                right_lane_id = lane_helpers[lane_idx + 1].lane_id if lane_idx < num_lanes - 1 else None
                left_lane_ids.append(left_lane_id)
                right_lane_ids.append(right_lane_id)

        data = pd.DataFrame(
            {
                "id": ids,
                "lane_group_id": lane_group_ids,
                "speed_limit_mps": speed_limits_mps,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
                "left_lane_id": left_lane_ids,
                "right_lane_id": right_lane_ids,
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
                "left_boundary": left_boundaries,
                "right_boundary": left_boundaries,
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

        for column in ["left_lane_id", "right_lane_id"]:
            lane_df[column] = lane_df[column].apply(
                lambda x: str(lane_id_mapping.str_to_int[x]) if pd.notna(x) and x is not None else x
            )

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
        road_edges = get_road_edges_3d_from_gdf(lane_df, carpark_df, generic_drivable_area_df, lane_group_df)
        road_edges = split_line_geometry_by_max_length(road_edges, MAX_ROAD_EDGE_LENGTH)

        ids = np.arange(len(road_edges), dtype=np.int64).tolist()
        # TODO @DanielDauner: Figure out if other types should/could be assigned here.
        road_edge_types = [int(RoadEdgeType.ROAD_EDGE_BOUNDARY)] * len(road_edges)
        geometries = road_edges
        return gpd.GeoDataFrame(pd.DataFrame({"id": ids, "road_edge_type": road_edge_types}), geometry=geometries)

    def _extract_road_line_df(
        self,
        lane_df: gpd.GeoDataFrame,
        lane_group_df: gpd.GeoDataFrame,
    ) -> None:

        lane_group_on_intersection = {
            lane_group_id: str(intersection_id) != "nan"
            for lane_group_id, intersection_id in zip(lane_group_df.id.tolist(), lane_group_df.intersection_id.tolist())
        }
        ids = []
        road_line_types = []
        geometries = []

        running_id = 0
        for lane_row in lane_df.itertuples():
            on_intersection = lane_group_on_intersection.get(lane_row.lane_group_id, False)
            if on_intersection:
                # Skip road lines on intersections
                continue
            if str(lane_row.right_lane_id) == "nan":
                # This is a boundary lane, e.g. a border or sidewalk
                ids.append(running_id)
                road_line_types.append(int(RoadLineType.SOLID_SINGLE_WHITE))
                geometries.append(lane_row.right_boundary)
                running_id += 1
            else:
                # This is a regular lane
                ids.append(running_id)
                road_line_types.append(int(RoadLineType.BROKEN_SINGLE_WHITE))
                geometries.append(lane_row.right_boundary)
                running_id += 1
            if str(lane_row.left_lane_id) == "nan":
                # This is a boundary lane, e.g. a border or sidewalk
                ids.append(running_id)
                road_line_types.append(int(RoadLineType.SOLID_SINGLE_WHITE))
                geometries.append(lane_row.left_boundary)
                running_id += 1

        data = pd.DataFrame({"id": ids, "road_line_type": road_line_types})
        return gpd.GeoDataFrame(data, geometry=geometries)


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
