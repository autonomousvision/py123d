from __future__ import annotations

from functools import cached_property
from typing import Dict, List, Optional, Tuple

import pandas as pd
from carl_nuplan.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_proximal_map_objects,
)
from carl_nuplan.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction_v2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import (
    Intersection,
    Lane,
    LaneConnector,
    PolygonMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.database.maps_db.map_api import NuPlanMap
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from shapely import Point, Polygon

from asim.simulation.gym.cache.helper.tracked_objects_filter import filter_tracked_objects
from asim.simulation.gym.environment.helper.environment_area import AbstractEnvironmentArea


class MapCache:
    """
    Helper class to save and load map-related data for the current environment area.
    NOTE: This class helps to avoid Map API calls during observation and reward computation.
    """

    def __init__(
        self,
        ego_state: EgoState,
        map_api: AbstractMap,
        environment_area: AbstractEnvironmentArea,
        traffic_light_status: List[TrafficLightStatusData],
        route_lane_group_ids: List[str],
        load_crosswalks: bool = False,
        load_stop_lines: bool = False,
        drivable_area_map: Optional[PDMOccupancyMap] = None,
    ) -> None:
        """
        Initializes the MapCache object.
        :param ego_state: Current ego state in the environment.
        :param map_api: Map interface of nuPlan maps.
        :param environment_area: Area to cache map data for.
        :param traffic_light_status: Current traffic light status data.
        :param route_lane_group_ids: List of lane group ids for the ego route.
        :param load_crosswalks: whether to load crosswalks, defaults to False
        :param load_stop_lines: whether to load stop lines, defaults to False
        :param drivable_area_map: Optional 2D occupancy map of drivable area objects, defaults to None
        """

        self.ego_state = ego_state
        self.map_api = map_api
        self.environment_area = environment_area
        self.load_crosswalks = load_crosswalks
        self.load_stop_lines = load_stop_lines

        self.route_lane_group_ids = route_lane_group_ids
        self.traffic_lights: Dict[str, TrafficLightStatusType] = {
            str(data.lane_connector_id): data.status for data in traffic_light_status
        }

        self.roadblocks: Dict[str, RoadBlockGraphEdgeMapObject] = {}
        self.lanes: Dict[str, Lane] = {}

        self.roadblock_connectors: Dict[str, RoadBlockGraphEdgeMapObject] = {}
        self.lane_connectors: Dict[str, LaneConnector] = {}

        self.intersections: Dict[str, Intersection] = {}
        self.stop_lines: Dict[str, StopLine] = {}
        self.car_parks: Dict[str, PolygonMapObject] = {}
        self.crosswalks: Dict[str, PolygonMapObject] = {}
        self._load_cache(drivable_area_map)

    def _load_cache(self, drivable_area_map: Optional[PDMOccupancyMap]) -> None:
        """
        Helper function to load the map cache during initialization.
        :param drivable_area_map: Optional 2D occupancy map of drivable area objects
        """

        if drivable_area_map is None:
            MAP_LAYERS = [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.ROADBLOCK_CONNECTOR,
                SemanticMapLayer.CARPARK_AREA,
            ]
            if self.load_stop_lines:
                MAP_LAYERS.append(SemanticMapLayer.STOP_LINE)
            if self.load_crosswalks:
                MAP_LAYERS.append(SemanticMapLayer.CROSSWALK)

            map_object_dict = get_proximal_map_objects(
                self.environment_area.get_global_polygon(self.ego_state.center),
                self.map_api,
                MAP_LAYERS,
            )
        else:
            patch = self.environment_area.get_global_polygon(self.ego_state.center)
            map_object_ids = drivable_area_map.intersects(patch)

            target_layers = [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.ROADBLOCK_CONNECTOR,
                SemanticMapLayer.CARPARK_AREA,
            ]
            map_object_dict = {layer: [] for layer in target_layers}
            for map_object_id in map_object_ids:
                for layer in target_layers:
                    map_object = self.map_api.get_map_object(map_object_id, layer=layer)
                    if map_object is not None:
                        map_object_dict[layer].append(map_object)
                        break

        for roadblock in map_object_dict[SemanticMapLayer.ROADBLOCK]:
            roadblock: RoadBlockGraphEdgeMapObject
            self.roadblocks[roadblock.id] = roadblock
            for lane in roadblock.interior_edges:
                self.lanes[lane.id] = lane

        for roadblock_connector in map_object_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]:
            roadblock_connector: RoadBlockGraphEdgeMapObject
            self.roadblock_connectors[roadblock_connector.id] = roadblock_connector

            optional_intersection: Intersection = roadblock_connector.intersection
            if optional_intersection is not None:
                self.intersections[optional_intersection.id] = optional_intersection

            for lane_connector in roadblock_connector.interior_edges:
                self.lane_connectors[lane_connector.id] = lane_connector

        for car_park in map_object_dict[SemanticMapLayer.CARPARK_AREA]:
            car_park: PolygonMapObject
            self.car_parks[car_park.id] = car_park

        if self.load_crosswalks:
            for crosswalk in map_object_dict[SemanticMapLayer.CROSSWALK]:
                crosswalk: PolygonMapObject
                self.crosswalks[crosswalk.id] = crosswalk

        if self.load_stop_lines:
            for stop_line in map_object_dict[SemanticMapLayer.STOP_LINE]:
                stop_line: StopLine
                self.stop_lines[stop_line.id] = stop_line

    @property
    def drivable_area_map(self) -> PDMOccupancyMap:
        """
        Returns a PDMOccupancyMap of the drivable area in the environment.
        :return: PDMOccupancyMap containing the drivable area layers (intersections, roadblocks, car parks).
        """
        tokens: List[str] = []
        polygons: List[Polygon] = []
        for element_dict in [self.intersections, self.roadblocks, self.car_parks]:
            for token, element in element_dict.items():
                tokens.append(token)
                polygons.append(element.polygon)
        return PDMOccupancyMap(tokens, polygons)

    @cached_property
    def origin(self) -> StateSE2:
        """
        Returns the global origin of the environment area based on the ego state.
        :return: Global origin of the environment area as StateSE2.
        """
        return self.environment_area.get_global_origin(self.ego_state.center)


class DetectionCache:
    """Helper class to save and load detection-related data for the current environment area."""

    def __init__(
        self, ego_state: EgoState, tracked_objects: TrackedObjects, environment_area: AbstractEnvironmentArea
    ) -> None:
        """
        Initializes the DetectionCache object.
        :param ego_state: Ego vehicle state in the environment.
        :param tracked_objects: Tracked objects wrapper of nuPlan.
        :param environment_area: Area to cache detection data for.
        """

        self.ego_state = ego_state
        self.environment_area = environment_area
        self.tracked_objects = tracked_objects

        self.vehicles: List[TrackedObject] = []
        self.pedestrians: List[TrackedObject] = []
        self.static_objects: List[TrackedObject] = []
        self._load_cache(tracked_objects)

    def _load_cache(self, tracked_objects: TrackedObjects):
        global_area_polygon = self.environment_area.get_global_polygon(self.ego_state.center)
        for tracked_object in tracked_objects.tracked_objects:
            if global_area_polygon.contains(Point(*tracked_object.center.array)):
                if tracked_object.tracked_object_type in [
                    TrackedObjectType.VEHICLE,
                    TrackedObjectType.BICYCLE,
                ]:
                    self.vehicles.append(tracked_object)
                elif tracked_object.tracked_object_type in [TrackedObjectType.PEDESTRIAN]:
                    self.pedestrians.append(tracked_object)
                elif tracked_object.tracked_object_type in [
                    TrackedObjectType.CZONE_SIGN,
                    TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT,
                ]:
                    self.static_objects.append(tracked_object)

    @cached_property
    def origin(self) -> StateSE2:
        """
        Returns the global origin of the environment area based on the ego state.
        :return: Global origin of the environment area as StateSE2.
        """
        return self.environment_area.get_global_origin(self.ego_state.center)


def build_environment_caches(
    planner_input: PlannerInput,
    planner_initialization: PlannerInitialization,
    environment_area: AbstractEnvironmentArea,
    route_lane_group_ids: Optional[List[str]] = None,
    route_correction: bool = False,
    track_filtering: bool = False,
) -> Tuple[MapCache, DetectionCache]:
    """
    Helper function to build the environment caches for the current planner input and initialization.
    :param planner_input: Planner input interface of nuPlan, ego, detection, and traffic light data.
    :param planner_initialization: Planner initialization interface of nuPlan, map API and route lane group ids.
    :param environment_area: Area object used to cache the map and detection data.
    :param route_lane_group_ids: Optional route lane group ids, to overwrite the planner initialization, defaults to None
    :param route_correction: Whether to apply route correction of lane group ids from planner initialization, defaults to False
    :param track_filtering: Whether to filter tracks by max counts based on the default configuration, defaults to False
    :return: Tuple of MapCache and DetectionCache objects.
    """

    ego_state, detection_tracks = planner_input.history.current_state
    assert isinstance(ego_state, EgoState)
    assert isinstance(detection_tracks, DetectionsTracks)

    if route_lane_group_ids is None:
        if route_correction:
            route_lane_group_ids = route_roadblock_correction_v2(
                ego_state, planner_initialization.map_api, planner_initialization.route_lane_group_ids
            )
        else:
            route_lane_group_ids = planner_initialization.route_lane_group_ids

    if track_filtering:
        tracked_objects = filter_tracked_objects(
            detection_tracks.tracked_objects, ego_state, planner_initialization.map_api
        )
    else:
        tracked_objects = detection_tracks.tracked_objects

    map_cache = MapCache(
        ego_state=ego_state,
        map_api=planner_initialization.map_api,
        environment_area=environment_area,
        traffic_light_status=list(planner_input.traffic_light_data),
        route_lane_group_ids=route_lane_group_ids,
    )
    detection_cache = DetectionCache(
        ego_state=ego_state, tracked_objects=tracked_objects, environment_area=environment_area
    )

    return map_cache, detection_cache


class EnvironmentCacheManager:
    """
    Helper function to improve performance of map api calls in the Gym environment.
    This class is not strictly necessary, but can have some performance benefits.
    NOTE: Class caches drivable areas of the four nuPlan maps as faster shapely STRtree.
    NOTE: Geopandas has own implantation of STRtree, but not provided in nuPlan interface.
    """

    def __init__(self) -> None:
        """
        Initializes the EnvironmentCacheManager object.
        """
        self._drivable_area_dfs: Dict[str, PDMOccupancyMap] = {}

    def _get_drivable_area_map(self, map_api: NuPlanMap) -> PDMOccupancyMap:
        """
        Helper to load the drivable area map for a given map API.
        :param map_api: Abstract map interface of nuPlan.
        :return: PDMOccupancyMap containing the drivable area layers (intersections, roadblocks, car parks).
        """
        if map_api.map_name not in self._drivable_area_dfs.keys():
            roadblocks = map_api._load_vector_map_layer("lane_groups_polygons")
            roadblock_connectors = map_api._load_vector_map_layer("lane_group_connectors")
            car_parks = map_api._load_vector_map_layer("carpark_areas")
            drivable_area_df = pd.concat([roadblocks, roadblock_connectors, car_parks]).dropna(axis=1, how="any")
            tokens = list(drivable_area_df["fid"])
            polygons = list(drivable_area_df["geometry"])
            self._drivable_area_dfs[map_api.map_name] = PDMOccupancyMap(tokens, polygons)

        return self._drivable_area_dfs[map_api.map_name]

    def build_environment_caches(
        self,
        planner_input: PlannerInput,
        planner_initialization: PlannerInitialization,
        environment_area: AbstractEnvironmentArea,
        route_roadblock_ids: Optional[List[str]] = None,
        route_correction: bool = False,
        track_filtering: bool = False,
    ) -> Tuple[MapCache, DetectionCache]:
        """
        Builds the environment caches for the current planner input and initialization.
        :param planner_input: Planner input interface of nuPlan, ego, detection, and traffic light data.
        :param planner_initialization: Planner initialization interface of nuPlan, map API and route roadblock ids.
        :param environment_area: Area object used to cache the map and detection data.
        :param route_roadblock_ids: Optional route roadblock ids, to overwrite the planner initialization, defaults to None
        :param route_correction: Whether to apply route correction of roadblock ids from planner initialization, defaults to False
        :param track_filtering: Whether to filter tracks by max counts based on the default configuration, defaults to False
        :return: Tuple of MapCache and DetectionCache objects.
        """

        ego_state, detection_tracks = planner_input.history.current_state
        assert isinstance(ego_state, EgoState)
        assert isinstance(detection_tracks, DetectionsTracks)
        drivable_area_map = self._get_drivable_area_map(planner_initialization.map_api)

        if route_roadblock_ids is None:
            if route_correction:
                route_roadblock_ids = route_roadblock_correction_v2(
                    ego_state,
                    planner_initialization.map_api,
                    planner_initialization.route_roadblock_ids,
                )
            else:
                route_roadblock_ids = planner_initialization.route_roadblock_ids

        if track_filtering:
            tracked_objects = filter_tracked_objects(
                detection_tracks.tracked_objects,
                ego_state,
                planner_initialization.map_api,
            )
        else:
            tracked_objects = detection_tracks.tracked_objects

        map_cache = MapCache(
            ego_state=ego_state,
            map_api=planner_initialization.map_api,
            environment_area=environment_area,
            traffic_light_status=list(planner_input.traffic_light_data),
            route_roadblock_ids=route_roadblock_ids,
            drivable_area_map=drivable_area_map,
        )
        detection_cache = DetectionCache(
            ego_state=ego_state,
            tracked_objects=tracked_objects,
            environment_area=environment_area,
        )

        return map_cache, detection_cache


environment_cache_manager = EnvironmentCacheManager()
