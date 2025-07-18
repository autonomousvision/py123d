from __future__ import annotations

import itertools
from enum import IntEnum
from functools import cached_property
from typing import Callable, Dict, Final, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_point_array,
    convert_relative_to_absolute_point_array,
)
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.transform import translate_longitudinally
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from shapely import LineString, Polygon, union_all, vectorized
from shapely.affinity import scale as shapely_scale

from asim.simulation.gym.environment.helper.environment_area import (
    AbstractEnvironmentArea,
    RectangleEnvironmentArea,
)
from asim.simulation.gym.environment.helper.environment_cache import DetectionCache, MapCache

# TODO: add to config
MIN_VALUE: Final[int] = 0  # Lowest value for a pixel in the raster
MAX_VALUE: Final[int] = 255  # Highest value for a pixel in the raster
LINE_THICKNESS: int = 1  # Width of the lines in pixels
TRAFFIC_LIGHT_VALUE: Dict[TrafficLightStatusType, int] = {
    TrafficLightStatusType.GREEN: 80,
    TrafficLightStatusType.YELLOW: 170,
    TrafficLightStatusType.RED: 255,
}
UNIONIZE: Final[bool] = False  # Whether to unionize polygons before rendering


class PolygonRenderType(IntEnum):
    CONVEX_SINGLE = 0
    NON_CONVEX_SINGLE = 1
    NON_CONVEX_BATCH = 2


RenderFunc = Callable[[npt.NDArray[np.uint8], List[npt.NDArray[np.int32]], int], None]


def _render_polygon_convex_single(
    raster: npt.NDArray[np.uint8], pixel_exteriors: List[npt.NDArray[np.int32]], color: int
) -> None:
    """
    Renders a list of convex polygons on the raster.
    :param raster: uint8 numpy array representing the raster to render on.
    :param pixel_exteriors: List of pixel exteriors of the polygons to render.
    :param color: Color to render the polygons in, as an integer value.
    """
    for pixel_exterior in pixel_exteriors:
        cv2.fillConvexPoly(raster, pixel_exterior, color=color)


def _render_polygon_non_convex_single(
    raster: npt.NDArray[np.uint8], pixel_exteriors: List[npt.NDArray[np.int32]], color: int
) -> None:
    """
    Renders a list of non-convex polygons on the raster.
    :param raster: uint8 numpy array representing the raster to render on.
    :param pixel_exteriors: List of pixel exteriors of the polygons to render.
    :param color: Color to render the polygons in, as an integer value.
    """
    for pixel_exterior in pixel_exteriors:
        cv2.fillPoly(raster, [pixel_exterior], color=color)


def _render_polygon_non_convex_batch(
    raster: npt.NDArray[np.uint8], pixel_exteriors: List[npt.NDArray[np.int32]], color: int
) -> None:
    """
    Renders a list of non-convex polygons on the raster batch-wise.
    :param raster: uint8 numpy array representing the raster to render on.
    :param pixel_exteriors: List of pixel exteriors of the polygons to render.
    :param color: Color to render the polygons in, as an integer value.
    """
    cv2.fillPoly(raster, pixel_exteriors, color=color)


POLYGON_RENDER_FUNCTIONS: Dict[PolygonRenderType, RenderFunc] = {
    PolygonRenderType.CONVEX_SINGLE: _render_polygon_convex_single,
    PolygonRenderType.NON_CONVEX_SINGLE: _render_polygon_non_convex_single,
    PolygonRenderType.NON_CONVEX_BATCH: _render_polygon_non_convex_batch,
}


def unionize_polygons(polygons: List[Polygon], grid_size: Optional[float] = None) -> List[Polygon]:
    """
    Unionizes a list of polygons into a single polygon or multiple polygons if they are disjoint.
    :param polygons: List of polygons to unionize.
    :param grid_size: Precision grid size for union call, defaults to None
    :return: List of polygon(s) after unionization.
    """
    unionized_polygons: List[Polygon] = []
    if len(polygons) == 1:
        unionized_polygons.append(polygons[0])
    elif len(polygons) > 1:
        union_polygon = union_all(polygons, grid_size=grid_size)
        if union_polygon.geom_type == "Polygon":
            unionized_polygons.append(union_polygon)
        elif union_polygon.geom_type == "MultiPolygon":
            for polygon in union_polygon.geoms:
                unionized_polygons.append(polygon)
    return unionized_polygons


class DefaultRenderer:
    """Renderer class for observation used in CaRL."""

    def __init__(
        self,
        environment_area: AbstractEnvironmentArea,
        pixel_per_meter: float = 2.0,
        max_vehicle_speed: float = 30.0,
        max_pedestrian_speed: float = 4.0,
        vehicle_scaling: float = 1.0,
        pedestrian_scaling: float = 1.0,
        static_scaling: float = 1.0,
        include_speed_line: bool = False,
        lane_connector_route: bool = False,
    ) -> None:
        """
        Initializes the DefaultRenderer object.
        :param environment_area: Area to render the observation in (should be rectangular).
        :param pixel_per_meter: number of pixels that should represent a meter in raster, defaults to 2.0
        :param max_vehicle_speed: Max vehicle speed after clipping for rendering the color, defaults to 30.0
        :param max_pedestrian_speed: Max pedestrian speed after clipping for rendering the color, defaults to 4.0
        :param vehicle_scaling: Factor to scale size of vehicle bounding boxes, defaults to 1.0
        :param pedestrian_scaling: Factor to scale size of pedestrian bounding boxes, defaults to 1.0
        :param static_scaling: Factor to scale size of static object bounding boxes, defaults to 1.0
        :param include_speed_line: Whether to include the constant velocity speed line into the raster, defaults to False
        :param lane_connector_route: Whether to use the lane connector (instead of roadblock connectors) for route, defaults to False
        """

        assert isinstance(
            environment_area, RectangleEnvironmentArea
        ), "DefaultRendering requires a rectangular environment area!"

        self._environment_area = environment_area
        self._pixel_per_meter = pixel_per_meter  # [ppm]

        self._max_vehicle_speed = max_vehicle_speed  # [m/s]
        self._max_pedestrian_speed = max_pedestrian_speed  # [m/s]

        self._vehicle_scaling = vehicle_scaling
        self._pedestrian_scaling = pedestrian_scaling
        self._static_scaling = static_scaling

        self._include_speed_line = include_speed_line

        # maybe remove:
        self._lane_connector_route = lane_connector_route
        self._polygon_render_type = PolygonRenderType.NON_CONVEX_SINGLE

    @cached_property
    def pixel_frame(self) -> Tuple[int, int]:
        """
        :return: Width and height of the pixel frame in pixels.
        """
        width, height = self._environment_area.frame
        return int(width * self._pixel_per_meter), int(height * self._pixel_per_meter)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        :return: Shape of the raster (including channel, width, height).
        """
        width, height = self.pixel_frame
        return (9, width, height)

    @property
    def _meter_per_pixel(self) -> float:
        """
        :return: Meters per pixel, i.e., the inverse of pixel_per_meter.
        """
        return 1 / self._pixel_per_meter

    def _get_global_pixel_centers(self, origin: StateSE2) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the pixel centers in the global coordinate system.
        :param origin: origin of the environment area in the global coordinate system.
        :return: Array of pixel centers in the global coordinate system.
        """
        width, height = self.pixel_frame
        boundary_x = ((width - 1) * self._meter_per_pixel) / 2
        boundary_y = ((height - 1) * self._meter_per_pixel) / 2
        x = np.arange(-boundary_x, boundary_x + self._meter_per_pixel, self._meter_per_pixel)
        y = np.arange(-boundary_y, boundary_y + self._meter_per_pixel, self._meter_per_pixel)
        local_pixel_centers = np.concatenate([ax[..., None] for ax in np.meshgrid(x, y)], axis=-1)
        return convert_relative_to_absolute_point_array(origin, local_pixel_centers)

    def _scale_to_color(self, value: Optional[float], max_value: float) -> int:
        """
        Scales a value to a color in the range [0, 255].
        :param value: Value to scale, if None, max_value is used instead.
        :param max_value: Maximum value to scale to color.
        :return: Scaled color value in the range [0, 255].
        """
        _value = value
        if value is None:
            _value = max_value
        normed = np.clip(_value / max_value, 0.0, 1.0)
        normed_color = np.clip(int((MAX_VALUE / 2) * normed + (MAX_VALUE / 2)), MIN_VALUE, MAX_VALUE)
        return int(normed_color)

    def _scale_polygon(self, polygon: Polygon, factor: float) -> Polygon:
        """
        Scales a polygon in size by a factor.
        :param polygon: shapely polygon to scale.
        :param factor: Scaling factor, e.g., 1.0 for no scaling, 0.5 for half size.
        :return: Scaled polygon.
        """
        if factor != 1.0:
            polygon = shapely_scale(polygon, xfact=factor, yfact=factor, origin="centroid")
        return polygon

    def _local_coords_to_pixel(self, coords: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
        """
        Converts local coordinates to pixel coordinates.
        :param coords: Local coordinates to convert, shape (N, 2).
        :return: Integer pixel coordinates, shape (N, 2).
        """
        pixel_width, pixel_height = self.pixel_frame
        pixel_center = np.array([[pixel_height, pixel_width]]) / 2.0
        pixel_coords = (coords * self._pixel_per_meter) + pixel_center
        return pixel_coords.astype(np.int32)

    def _global_coords_to_pixel(self, origin: StateSE2, coords: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """
        Converts global coordinates to pixel coordinates.
        :param origin: SE2 of origin, i.e. the center of the raster in global coordinates.
        :param coords: Global coordinates to convert, shape (N, 2).
        :return: Integer pixel coordinates, shape (N, 2).
        """
        local_coords = convert_absolute_to_relative_point_array(origin, coords)
        return self._local_coords_to_pixel(local_coords)

    def _global_polygon_to_pixel(self, origin: StateSE2, polygon: Polygon) -> npt.NDArray[np.int32]:
        """
        Converts a global polygon to pixel coordinates.
        :param origin: SE2 of origin, i.e. the center of the raster in global coordinates.
        :param polygon: Shapely polygon to convert.
        :return: Integer pixel coordinates of the polygon exterior, shape (N, 1, 2).
        """
        exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
        return self._global_coords_to_pixel(origin, exterior)

    def _global_linestring_to_pixel(self, origin: StateSE2, linestring: LineString) -> npt.NDArray[np.int32]:
        """
        Converts a global linestring to pixel coordinates.
        :param origin: SE2 of origin, i.e. the center of the raster in global coordinates.
        :param linestring: Shapely linestring to convert.
        :return: Integer pixel coordinates of the linestring, shape (N, 1, 2).
        """
        coords = np.array(linestring.coords).reshape((-1, 1, 2))
        return self._global_coords_to_pixel(origin, coords)

    def _render_polygons(
        self,
        raster: npt.NDArray[np.uint8],
        origin: StateSE2,
        polygons: List[Polygon],
        color: int = MAX_VALUE,
    ) -> None:
        """
        Renders a list of arbitrary polygons on the raster.
        :param raster: uint8 numpy array representing the raster to render on.
        :param origin: SE2 of origin, i.e. the center of the raster in global coordinates.
        :param polygons: List of shapely polygons to render.
        :param color: Integer value of color, defaults to MAX_VALUE
        """
        if len(polygons) > 0:
            pixel_exteriors: List[npt.NDArray[np.int32]] = []
            if UNIONIZE:
                polygons = unionize_polygons(polygons, grid_size=None)

            for polygon in polygons:
                pixel_exteriors.append(self._global_polygon_to_pixel(origin, polygon))
            POLYGON_RENDER_FUNCTIONS[PolygonRenderType.NON_CONVEX_SINGLE](raster, pixel_exteriors, color)

    def _render_convex_polygons(
        self,
        raster: npt.NDArray[np.uint8],
        origin: StateSE2,
        polygons: List[Polygon],
        color: int = MAX_VALUE,
    ) -> None:
        """
        Renders a list of convex polygons on the raster.
        :param raster: uint8 numpy array representing the raster to render on.
        :param origin: SE2 of origin, i.e. the center of the raster in global coordinates.
        :param polygons: List of shapely polygons to render.
        :param color: Integer value of color, defaults to MAX_VALUE
        """
        if len(polygons) > 0:
            pixel_exteriors: List[npt.NDArray[np.int32]] = []
            for polygon in polygons:
                pixel_exteriors.append(self._global_polygon_to_pixel(origin, polygon))
            POLYGON_RENDER_FUNCTIONS[PolygonRenderType.CONVEX_SINGLE](raster, pixel_exteriors, color)

    def _render_pip_polygons(
        self,
        raster: npt.NDArray[np.uint8],
        global_pixel_centers: npt.NDArray[np.float64],
        polygons: List[Polygon],
        color: int = MAX_VALUE,
    ) -> None:
        """
        Renders polygons on the raster using a point-in-polygon approach.
        NOTE: This is not efficient. Function should be removed.
        :param raster: uint8 numpy array representing the raster to render on.
        :param global_pixel_centers: Global pixel centers in the raster, shape (H, W).
        :param polygons: List of shapely polygons to render.
        :param color: Integer value of color, defaults to MAX_VALUE
        """
        if len(polygons) > 0:
            flat_pixel_centers = global_pixel_centers.reshape(-1, 2)
            in_polygon_mask = np.zeros((len(polygons), len(flat_pixel_centers)), dtype=bool)
            for polygon_idx, polygon in enumerate(polygons):
                in_polygon_mask[polygon_idx] = vectorized.contains(
                    polygon, flat_pixel_centers[..., 0], flat_pixel_centers[..., 1]
                )
            in_polygon_mask = in_polygon_mask.any(axis=0)
            in_polygon_mask = in_polygon_mask.reshape(self.pixel_frame)
            raster[in_polygon_mask] = color

    def _render_linestrings(
        self,
        raster: npt.NDArray[np.uint8],
        origin: StateSE2,
        linestrings: List[LineString],
        color: int = MAX_VALUE,
    ) -> None:
        """
        Renders a list of linestrings on the raster.
        :param raster: uint8 numpy array representing the raster to render on.
        :param origin: SE2 of origin, i.e. the center of the raster in global coordinates.
        :param linestrings: List of shapely linestrings to render.
        :param color: Integer value of color, defaults to MAX_VALUE
        """
        if len(linestrings) > 0:
            pixel_linestrings: List[npt.NDArray[np.int32]] = []
            for linestring in linestrings:
                pixel_linestrings.append(self._global_linestring_to_pixel(origin, linestring))
            cv2.polylines(
                raster,
                pixel_linestrings,
                isClosed=False,
                color=color,
                thickness=LINE_THICKNESS,
            )

    def _render_speed_line(self, raster: npt.NDArray[np.uint8], origin: StateSE2, agent: Agent, color: int) -> None:
        """
        Renders a speed line for the agent on the raster.
        :param raster: uint8 numpy array representing the raster to render on.
        :param origin: SE2 of origin, i.e. the center of the raster in global coordinates.
        :param agent: Agent object containing the state and velocity.
        :param color: Integer value of color
        """
        if agent.velocity.magnitude() > self._meter_per_pixel:
            future = translate_longitudinally(
                agent.box.center,
                distance=agent.box.half_length + agent.velocity.magnitude(),  # * self._pixel_per_meter,
            )
            linestring = LineString(
                [
                    [agent.box.center.x, agent.box.center.y],
                    [future.x, future.y],
                ]
            )
            self._render_linestrings(raster, origin, [linestring], color=color)

    def _get_empty_raster(self) -> npt.NDArray[np.uint8]:
        """
        Helper function to create an empty raster with the shape of the pixel frame.
        :return: Empty raster with the shape of the pixel frame.
        """
        pixel_width, pixel_height = self.pixel_frame
        return np.zeros((pixel_width, pixel_height), dtype=np.uint8)

    def _render_map_from_cache(self, map_cache: MapCache) -> List[npt.NDArray[np.uint8]]:
        """
        Renders the map from the map cache into a list of rasters.
        :param map_cache: MapCache object containing the map data.
        :return: List of rasters representing the map data.
        """

        # 1. Drivable Area (Roadblock, Intersection, Car-Park), Polygon
        # 2. Route (Roadblock, Roadblock-Connector), Polygon
        # 3. Lane Boundaries (Lane), Polyline
        # 6. Traffic Light (Lane-Connector), Polygon
        # 7. Stop-Signs (Stop-Signs), Polygon
        # 8. Speed-Signs (Lane, Lane-Connector), Polygon
        drivable_area_raster = self._get_empty_raster()
        route_raster = self._get_empty_raster()
        lane_boundary_raster = self._get_empty_raster()
        traffic_light_raster = self._get_empty_raster()
        stop_sign_raster = self._get_empty_raster()
        speed_raster = self._get_empty_raster()

        mask = self._get_empty_raster()
        drivable_area_polygons: List[Polygon] = []
        route_polygons: List[Polygon] = []
        stop_sign_polygons: List[Polygon] = []
        lane_boundary_linestrings: List[LineString] = []

        for roadblock_id, roadblock in map_cache.roadblocks.items():
            # Roadblock: (1) drivable_area_raster, (2) route_raster
            self._render_polygons(mask, map_cache.origin, [roadblock.polygon], color=MAX_VALUE)
            drivable_area_raster[mask == MAX_VALUE] = MAX_VALUE
            if roadblock_id in map_cache.route_roadblock_ids:
                route_raster[mask == MAX_VALUE] = MAX_VALUE
            mask.fill(0)

        for (
            roadblock_connector_id,
            roadblock_connector,
        ) in map_cache.roadblock_connectors.items():
            # RoadblockConnector: (2) route_raster
            if roadblock_connector_id in map_cache.route_roadblock_ids:
                if self._lane_connector_route:
                    route_polygons.extend(
                        unionize_polygons([lane.polygon for lane in roadblock_connector.interior_edges])
                    )
                else:
                    route_polygons.append(roadblock_connector.polygon)

        for lane in map_cache.lanes.values():
            # Lane: (3) lane_boundary_raster, (6) speed_raster
            # - (3) lane_boundary_raster
            lane_boundary_linestrings.extend([lane.right_boundary.linestring, lane.left_boundary.linestring])

            # (6) speed_raster
            self._render_linestrings(
                speed_raster,
                map_cache.origin,
                [lane.baseline_path.linestring],
                color=self._scale_to_color(lane.speed_limit_mps, self._max_vehicle_speed),
            )

        for lane_connector_id, lane_connector in map_cache.lane_connectors.items():
            # Lane: (4) traffic_light_raster, (6) speed_raster, [optional: (2) route_raster]
            self._render_linestrings(
                mask,
                map_cache.origin,
                [lane_connector.baseline_path.linestring],
                color=MAX_VALUE,
            )

            # (4) traffic_light_raster
            if lane_connector_id in map_cache.traffic_lights.keys():
                traffic_light_status_type = map_cache.traffic_lights[lane_connector_id]
                traffic_light_raster[mask == MAX_VALUE] = TRAFFIC_LIGHT_VALUE[traffic_light_status_type]

            # (6) speed_raster
            speed_raster[mask == MAX_VALUE] = self._scale_to_color(
                lane_connector.speed_limit_mps, self._max_vehicle_speed
            )
            mask.fill(0)

        for drivable_area_element in itertools.chain(map_cache.intersections.values(), map_cache.car_parks.values()):
            # Intersections & Carparks: (1) drivable_area_raster
            drivable_area_polygons.append(drivable_area_element.polygon)

        for stop_sign in map_cache.stop_lines.values():
            # Stop Signs: (1) stop_sign_raster
            stop_sign_polygons.append(stop_sign.polygon)

        self._render_polygons(drivable_area_raster, map_cache.origin, drivable_area_polygons, color=MAX_VALUE)
        self._render_polygons(route_raster, map_cache.origin, route_polygons, color=MAX_VALUE)
        self._render_polygons(stop_sign_raster, map_cache.origin, stop_sign_polygons, color=MAX_VALUE)
        self._render_linestrings(lane_boundary_raster, map_cache.origin, lane_boundary_linestrings, color=MAX_VALUE)

        return [
            drivable_area_raster,
            route_raster,
            lane_boundary_raster,
            traffic_light_raster,
            stop_sign_raster,
            speed_raster,
        ]

    def _render_detections_from_cache(self, detection_cache: DetectionCache) -> List[npt.NDArray[np.uint8]]:
        """
        Renders the detections from the detection cache into a list of rasters.
        :param detection_cache: DetectionCache object containing the detection data.
        :return: List of rasters representing the detection data.
        """

        mask = self._get_empty_raster()

        # 1. Vehicles (Vehicles, Bicycles), Polygon, LineString
        # 2. Pedestrians+Static (Pedestrians, Static objects), Polygon
        vehicles_raster = self._get_empty_raster()
        pedestrians_raster = self._get_empty_raster()
        ego_raster = self._get_empty_raster()

        # 1. Vehicles
        for vehicle in detection_cache.vehicles:
            if self._include_speed_line:
                self._render_speed_line(mask, detection_cache.origin, vehicle, color=MAX_VALUE)

            polygon: Polygon = self._scale_polygon(vehicle.box.geometry, self._vehicle_scaling)
            self._render_convex_polygons(mask, detection_cache.origin, [polygon], color=MAX_VALUE)
            vehicles_raster[mask > 0] = self._scale_to_color(
                vehicle.velocity.magnitude(),
                self._max_vehicle_speed,
            )
            mask.fill(0)

        # 2. Pedestrian
        for pedestrian in detection_cache.pedestrians:
            if self._include_speed_line:
                self._render_speed_line(mask, detection_cache.origin, pedestrian, color=MAX_VALUE)

            polygon: Polygon = self._scale_polygon(pedestrian.box.geometry, self._pedestrian_scaling)
            self._render_convex_polygons(mask, detection_cache.origin, [polygon], color=MAX_VALUE)
            pedestrians_raster[mask > 0] = self._scale_to_color(
                pedestrian.velocity.magnitude(),
                self._max_pedestrian_speed,
            )
            mask.fill(0)

        # 3. Static Objects
        static_polygons: List[Polygon] = []
        for static_object in detection_cache.static_objects:
            polygon: Polygon = self._scale_polygon(static_object.box.geometry, self._static_scaling)
            static_polygons.append(polygon)
        self._render_convex_polygons(
            pedestrians_raster,
            detection_cache.origin,
            static_polygons,
            color=self._scale_to_color(0.0, self._max_vehicle_speed),
        )

        # 4. Ego Vehicle
        ego_agent = detection_cache.ego_state.agent
        if self._include_speed_line:
            self._render_speed_line(mask, detection_cache.origin, ego_agent, color=MAX_VALUE)

        ego_polygon: Polygon = self._scale_polygon(ego_agent.box.geometry, self._vehicle_scaling)
        self._render_convex_polygons(mask, detection_cache.origin, [ego_polygon], color=MAX_VALUE)
        ego_raster[mask > 0] = self._scale_to_color(
            ego_agent.velocity.magnitude(),
            self._max_vehicle_speed,
        )
        mask.fill(0)

        return [vehicles_raster, pedestrians_raster, ego_raster]

    def render(self, map_cache: MapCache, detection_cache: DetectionCache) -> npt.NDArray[np.uint8]:
        """
        Renders the map and detections from the caches into a single raster.
        :param map_cache: MapCache object containing the map data.
        :param detection_cache: DetectionCache object containing the detection data.
        :return: Raster representing the map and detections.
        """
        map_raster = self._render_map_from_cache(map_cache)
        detection_raster = self._render_detections_from_cache(detection_cache)

        raster: npt.NDArray[np.uint8] = np.concatenate(
            [channel[None, ...] for channel in (map_raster + detection_raster)], axis=0
        )
        return raster
