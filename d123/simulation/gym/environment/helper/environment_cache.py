from __future__ import annotations

from functools import cached_property
from typing import Dict, List, Optional, Tuple

from shapely import Polygon

from d123.common.datatypes.detection.detection import (
    BoxDetectionSE2,
    BoxDetectionWrapper,
    TrafficLightDetectionWrapper,
    TrafficLightStatus,
)
from d123.common.datatypes.detection.detection_types import DetectionType
from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.common.geometry.base import StateSE2
from d123.common.geometry.occupancy_map import OccupancyMap2D
from d123.dataset.maps.abstract_map import AbstractMap
from d123.dataset.maps.abstract_map_objects import (
    AbstractCarpark,
    AbstractCrosswalk,
    AbstractIntersection,
    AbstractLane,
    AbstractLaneGroup,
    AbstractStopLine,
)
from d123.dataset.maps.map_datatypes import MapSurfaceType
from d123.simulation.gym.environment.helper.environment_area import AbstractEnvironmentArea
from d123.simulation.planning.abstract_planner import PlannerInitialization, PlannerInput


class MapCache:
    """
    Helper class to save and load map-related data for the current environment area.
    NOTE: This class helps to avoid Map API calls during observation and reward computation.
    """

    def __init__(
        self,
        ego_state: EgoStateSE2,
        map_api: AbstractMap,
        environment_area: AbstractEnvironmentArea,
        traffic_lights: TrafficLightDetectionWrapper,
        route_lane_group_ids: List[str],
        load_crosswalks: bool = False,
        load_stop_lines: bool = False,
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
        """

        self.ego_state = ego_state
        self.map_api = map_api
        self.environment_area = environment_area
        self.load_crosswalks = load_crosswalks
        self.load_stop_lines = load_stop_lines

        self.route_lane_group_ids = route_lane_group_ids
        self.traffic_lights: Dict[str, TrafficLightStatus] = {str(data.lane_id): data.status for data in traffic_lights}

        self.lane_groups: Dict[str, AbstractLaneGroup] = {}
        self.lanes: Dict[str, AbstractLane] = {}

        self.intersections: Dict[str, AbstractIntersection] = {}
        self.stop_lines: Dict[str, AbstractStopLine] = {}
        self.car_parks: Dict[str, AbstractCarpark] = {}
        self.crosswalks: Dict[str, AbstractCrosswalk] = {}
        self._load_cache()

    def _load_cache(self) -> None:

        query_map_layers = [MapSurfaceType.LANE_GROUP, MapSurfaceType.CARPARK]
        # FIXME: Add stop lines and crosswalks to the map layers if needed
        # if self.load_crosswalks:
        #     query_map_layers.append(MapSurfaceType.CROSSWALK)
        # if self.load_stop_lines:
        #     query_map_layers.append(MapSurfaceType.STOP_LINE)

        map_object_dict = self.map_api.query(
            geometry=self.environment_area.get_global_polygon(self.ego_state.center),
            layers=query_map_layers,
            predicate="intersects",
        )

        # 1. load (1.1) lane groups, (1.2) lanes, (1.3) intersections
        for lane_group in map_object_dict[MapSurfaceType.LANE_GROUP]:
            lane_group: AbstractLaneGroup
            self.lane_groups[lane_group.id] = lane_group
            for lane in lane_group.lanes:
                self.lanes[lane.id] = lane
            optional_intersection = lane_group.intersection
            if optional_intersection is not None:
                self.intersections[optional_intersection.id] = optional_intersection

        # 2. load car parks
        for car_park in map_object_dict[MapSurfaceType.CARPARK]:
            car_park: AbstractCarpark
            self.car_parks[car_park.id] = car_park

        # FIXME: Add stop lines and crosswalks to the map layers if needed
        # if self.load_crosswalks:
        #     for crosswalk in map_object_dict[MapSurfaceType.CROSSWALK]:
        #         crosswalk: AbstractCarpark
        #         self.crosswalks[crosswalk.id] = crosswalk

        # if self.load_stop_lines:
        #     for stop_line in map_object_dict[MapSurfaceType.STOP_LINE]:
        #         stop_line: AbstractStopLine
        #         self.stop_lines[stop_line.id] = stop_line

    @property
    def drivable_area_map(self) -> OccupancyMap2D:

        tokens: List[str] = []
        polygons: List[Polygon] = []

        # FIXME: Remove lane groups on intersections
        for element_dict in [self.intersections, self.lane_groups, self.car_parks]:
            for token, element in element_dict.items():
                tokens.append(token)
                polygons.append(element.polygon)
        return OccupancyMap2D(polygons, tokens)

    @cached_property
    def origin(self) -> StateSE2:
        """
        Returns the global origin of the environment area based on the ego state.
        :return: Global origin of the environment area as StateSE2.
        """
        return self.environment_area.get_global_origin(self.ego_state.center)


class BoxDetectionCache:
    """Helper class to save and load detection-related data for the current environment area."""

    def __init__(
        self,
        ego_state: EgoStateSE2,
        box_detections: BoxDetectionWrapper,
        environment_area: AbstractEnvironmentArea,
    ) -> None:
        """
        Initializes the BoxDetectionCache object.
        :param ego_state: Ego vehicle state in the environment.
        :param tracked_objects: Tracked objects wrapper of nuPlan.
        :param environment_area: Area to cache detection data for.
        """

        self.ego_state = ego_state
        self.environment_area = environment_area
        self.tracked_objects = box_detections

        self.vehicles: List[BoxDetectionSE2] = []
        self.pedestrians: List[BoxDetectionSE2] = []
        self.static_objects: List[BoxDetectionSE2] = []
        self._load_cache(box_detections)

    def _load_cache(self, box_detections: BoxDetectionWrapper) -> None:
        global_area_polygon = self.environment_area.get_global_polygon(self.ego_state.center)

        for box_detection in box_detections:
            if global_area_polygon.contains(box_detection.center.shapely_point):
                if box_detection.metadata.detection_type in [DetectionType.VEHICLE, DetectionType.BICYCLE]:
                    self.vehicles.append(box_detection)
                elif box_detection.metadata.detection_type in [DetectionType.PEDESTRIAN]:
                    self.pedestrians.append(box_detection)
                elif box_detection.metadata.detection_type in [
                    DetectionType.CZONE_SIGN,
                    DetectionType.BARRIER,
                    DetectionType.TRAFFIC_CONE,
                    DetectionType.GENERIC_OBJECT,
                ]:
                    self.static_objects.append(box_detection)

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
) -> Tuple[MapCache, BoxDetectionCache]:
    """
    Helper function to build the environment caches for the current planner input and initialization.
    :param planner_input: Planner input interface of nuPlan, ego, detection, and traffic light data.
    :param planner_initialization: Planner initialization interface of nuPlan, map API and route lane group ids.
    :param environment_area: Area object used to cache the map and detection data.
    :param route_lane_group_ids: Optional route lane group ids, to overwrite the planner initialization, defaults to None
    :return: Tuple of MapCache and DetectionCache objects.
    """

    ego_state, detection_recording = planner_input.history.current_state
    assert isinstance(detection_recording, DetectionRecording), "Recording must be of type DetectionRecording"

    # TODO: Add route correction?
    route_lane_group_ids = planner_initialization.route_lane_group_ids

    # TODO: Add box detection filtering?
    box_detections = detection_recording.box_detections

    map_cache = MapCache(
        ego_state=ego_state,
        map_api=planner_initialization.map_api,
        environment_area=environment_area,
        traffic_lights=detection_recording.traffic_light_detections,
        route_lane_group_ids=route_lane_group_ids,
    )
    detection_cache = BoxDetectionCache(
        ego_state=ego_state,
        box_detections=box_detections,
        environment_area=environment_area,
    )

    return map_cache, detection_cache
