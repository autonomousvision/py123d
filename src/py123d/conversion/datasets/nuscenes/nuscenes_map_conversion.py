import lanelet2
import numpy as np

from pathlib import Path
from typing import List, Optional
from lanelet2.io import load
from lanelet2.projection import MercatorProjector
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.validation import make_valid
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.arcline_path_utils import discretize_lane

from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.datatypes.maps.cache.cache_map_objects import (
    CacheCarpark,
    CacheCrosswalk,
    CacheGenericDrivable,
    CacheIntersection,
    CacheLane,
    CacheLaneGroup,
    CacheRoadLine,
    CacheWalkway,
)
from py123d.datatypes.maps.map_datatypes import RoadLineType
from py123d.geometry import Polyline3D

NUSCENES_MAPS: List[str] = [
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown"
]


def write_nuscenes_map(nuscenes_maps_root: Path, location: str, map_writer: AbstractMapWriter, use_lanelet2: bool, lanelet2_root: Optional[str] = None) -> None:
    """
    Main function to convert nuscenes map to unified format and write using map_writer.
    """
    assert location in NUSCENES_MAPS, f"Map name {location} is not supported."
    nusc_map = NuScenesMap(dataroot=str(nuscenes_maps_root), map_name=location)
    
    # Write all layers
    if use_lanelet2:
        _write_nuscenes_lanes_lanelet2(nusc_map, map_writer, lanelet2_root)
        _write_nuscenes_lane_groups_lanelet2(nusc_map, map_writer, lanelet2_root)
    else:
        _write_nuscenes_lanes(nusc_map, map_writer)
        _write_nuscenes_lane_groups(nusc_map, map_writer)
    _write_nuscenes_intersections(nusc_map, map_writer)
    _write_nuscenes_crosswalks(nusc_map, map_writer)
    _write_nuscenes_walkways(nusc_map, map_writer)
    _write_nuscenes_carparks(nusc_map, map_writer)
    _write_nuscenes_generic_drivables(nusc_map, map_writer)
    _write_nuscenes_stop_lines(nusc_map, map_writer)
    _write_nuscenes_road_lines(nusc_map, map_writer)


def _write_nuscenes_lanes_lanelet2(nusc_map: NuScenesMap, map_writer: AbstractMapWriter, lanelet2_root: str) -> None:
    map_name = nusc_map.map_name
    osm_map_file = str(Path(lanelet2_root) / f"{map_name}.osm")
    
    if "boston" in map_name.lower():
        origin_lat, origin_lon = 42.3365, -71.0577
    elif "singapore" in map_name.lower():
        origin_lat, origin_lon = 1.3, 103.8
    else:
        origin_lat, origin_lon = 49.0, 8.4
    
    origin = lanelet2.io.Origin(origin_lat, origin_lon)
    
    try:
        lanelet_map = lanelet2.io.load(osm_map_file, origin)
    except Exception:
        try:
            projector = lanelet2.projection.MercatorProjector(origin)
            lanelet_map = lanelet2.io.load(osm_map_file, projector)
        except Exception:
            return

    for lanelet in lanelet_map.laneletLayer:
        token = lanelet.id
        
        try:
            left_bound = [(p.x, p.y) for p in lanelet.leftBound]
            right_bound = [(p.x, p.y) for p in lanelet.rightBound]
            polygon_points = left_bound + right_bound[::-1]
            polygon = Polygon(polygon_points)
            
            predecessor_ids = [int(pred.id) for pred in lanelet.previousLanelets]
            successor_ids = [int(succ.id) for succ in lanelet.followingLanelets]
            
            left_lane_id = None
            right_lane_id = None
            
            left_boundary = [(p.x, p.y) for p in lanelet.leftBound]
            right_boundary = [(p.x, p.y) for p in lanelet.rightBound]
            centerline = []
            for left_pt, right_pt in zip(lanelet.leftBound, lanelet.rightBound):
                center_x = (left_pt.x + right_pt.x) / 2
                center_y = (left_pt.y + right_pt.y) / 2
                centerline.append((center_x, center_y))
            
            speed_limit_mps = 0.0
            if "speed_limit" in lanelet.attributes:
                try:
                    speed_limit_str = lanelet.attributes["speed_limit"]
                    if "km/h" in speed_limit_str:
                        speed_kmh = float(speed_limit_str.replace("km/h", "").strip())
                        speed_limit_mps = speed_kmh / 3.6
                except (ValueError, TypeError):
                    pass
            
            map_writer.write_lane(
                CacheLane(
                    object_id=token,
                    lane_group_id=None,
                    left_boundary=left_boundary,
                    right_boundary=right_boundary,
                    centerline=centerline,
                    left_lane_id=left_lane_id,
                    right_lane_id=right_lane_id,
                    predecessor_ids=predecessor_ids,
                    successor_ids=successor_ids,
                    speed_limit_mps=speed_limit_mps,
                    outline=None,
                    geometry=polygon,
                )
            )
        except Exception:
            continue

def _write_nuscenes_lane_groups_lanelet2(nusc_map: NuScenesMap, map_writer: AbstractMapWriter, lanelet2_root: str) -> None:
    map_name = nusc_map.map_name
    osm_map_file = str(Path(lanelet2_root) / f"{map_name}.osm")

    if "boston" in map_name.lower():
        origin_lat, origin_lon = 42.3365, -71.0577
    else:
        origin_lat, origin_lon = 1.3, 103.8
    
    origin = lanelet2.io.Origin(origin_lat, origin_lon)
    
    try:
        projector = MercatorProjector(origin)
        lanelet_map = load(osm_map_file, projector)
    except Exception:
        return

    for lanelet in lanelet_map.laneletLayer:
        token = lanelet.id
        lane_ids = [lanelet.id]
        try:
            predecessor_ids = [int(lanelet.id) for lanelet in lanelet.previous]
            successor_ids = [int(lanelet.id) for lanelet in lanelet.following]
        except AttributeError:
            predecessor_ids = []
            successor_ids = []
            try:
                if hasattr(lanelet, 'left'):
                    for left_lane in lanelet.left:
                        predecessor_ids.append(int(left_lane.id))
                if hasattr(lanelet, 'right'):
                    for right_lane in lanelet.right:
                        successor_ids.append(int(right_lane.id))
            except Exception:
                pass

        try:
            left_bound = [(p.x, p.y) for p in lanelet.leftBound]
            right_bound = [(p.x, p.y) for p in lanelet.rightBound]
            polygon_points = left_bound + right_bound[::-1]
            polygon = Polygon(polygon_points)
        except Exception:
            continue

        try:
            map_writer.write_lane_group(
                CacheLaneGroup(
                    object_id=token,
                    lane_ids=lane_ids,
                    left_boundary=None,
                    right_boundary=None,
                    intersection_id=None,
                    predecessor_ids=predecessor_ids,
                    successor_ids=successor_ids,
                    outline=None,
                    geometry=polygon,
                )
            )
        except Exception:
            continue

def _get_lanelet_connections(lanelet):
    """
    Helper function to extract incoming and outgoing lanelets.
    """
    incoming = lanelet.incomings
    outgoing = lanelet.outgoings
    return incoming, outgoing


def _write_nuscenes_lanes(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write lane data to map_writer, including topology and boundaries.
    """
    lane_records = nusc_map.lane
    for lane_record in lane_records:
        token = lane_record["token"]

        # Extract geometry from lane record
        try:
            if "polygon_token" in lane_record:
                polygon = nusc_map.extract_polygon(lane_record["polygon_token"])
            else:
                continue
            if not polygon.is_valid:
                continue
        except Exception:
            continue

        # Get topology
        incoming = nusc_map.get_incoming_lane_ids(token)
        outgoing = nusc_map.get_outgoing_lane_ids(token)

        # Get lane connectors
        lane_connectors = []
        for connector in nusc_map.lane_connector:
            if connector.get("incoming_lane") == token or connector.get("outgoing_lane") == token:
                lane_connectors.append(connector["token"])

        # Extract boundaries
        left_boundary = _get_lane_boundary(token, "left", nusc_map)
        right_boundary = _get_lane_boundary(token, "right", nusc_map)

        # Skip lanes without valid boundaries
        if left_boundary is None or right_boundary is None:
            continue
        if left_boundary.is_empty or right_boundary.is_empty:
            continue

        # Extract baseline path
        baseline_path = None
        if token in nusc_map.arcline_path_3:
            arc_path = nusc_map.arcline_path_3[token]
            try:
                points = discretize_lane(arc_path, resolution_meters=0.1)
                xy_points = [(p[0], p[1]) for p in points]
                baseline_path = LineString(xy_points)
            except Exception:
                baseline_path = None

        # Align boundaries with baseline path direction
        if baseline_path and left_boundary:
            left_boundary = align_boundary_direction(baseline_path, left_boundary)
        if baseline_path and right_boundary:
            right_boundary = align_boundary_direction(baseline_path, right_boundary)

        # Write lane object safely
        try:
            map_writer.write_lane(
                CacheLane(
                    object_id=token,
                    lane_group_id=lane_record.get("road_segment_token", None),
                    left_boundary=left_boundary,
                    right_boundary=right_boundary,
                    centerline=baseline_path,
                    left_lane_id=None,  # Not directly available in nuscenes
                    right_lane_id=None,  # Not directly available in nuscenes
                    predecessor_ids=incoming,
                    successor_ids=outgoing,
                    speed_limit_mps=0.0,  # Default value
                    outline=None,
                    geometry=polygon,
                )
            )
        except Exception:
            continue


def _write_nuscenes_lane_groups(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write lane group data to map_writer.
    """
    road_segments = nusc_map.road_segment
    for segment in road_segments:
        token = segment["token"]

        # Extract geometry
        try:
            if "polygon_token" in segment:
                polygon = nusc_map.extract_polygon(segment["polygon_token"])
            else:
                continue
            if not polygon.is_valid:
                continue
        except Exception:
            continue

        # Find lanes in this segment
        lane_ids = []
        for lane in nusc_map.lane:
            if lane.get("road_segment_token") == token:
                lane_ids.append(lane["token"])

        # Get connected segments
        incoming, outgoing = _get_connected_segments(token, nusc_map)

        # Extract boundaries
        left_boundary = _get_lane_group_boundary(token, "left", nusc_map)
        right_boundary = _get_lane_group_boundary(token, "right", nusc_map)

        # Skip invalid boundaries
        if left_boundary is None or right_boundary is None:
            continue
        if left_boundary.is_empty or right_boundary.is_empty:
            continue

        # Use first lane's baseline path for direction alignment
        baseline_path = None
        if lane_ids:
            first_lane_token = lane_ids[0]
            if first_lane_token in nusc_map.arcline_path_3:
                arc_path = nusc_map.arcline_path_3[first_lane_token]
                try:
                    points = discretize_lane(arc_path, resolution_meters=0.1)
                    xy_points = [(p[0], p[1]) for p in points]
                    baseline_path = LineString(xy_points)
                except Exception:
                    baseline_path = None

        if baseline_path and left_boundary:
            left_boundary = align_boundary_direction(baseline_path, left_boundary)
        if baseline_path and right_boundary:
            right_boundary = align_boundary_direction(baseline_path, right_boundary)

        try:
            map_writer.write_lane_group(
                CacheLaneGroup(
                    object_id=token,
                    lane_ids=lane_ids,
                    left_boundary=left_boundary,
                    right_boundary=right_boundary,
                    intersection_id=None,  # Handled in intersections
                    predecessor_ids=incoming,
                    successor_ids=outgoing,
                    outline=None,
                    geometry=polygon,
                )
            )
        except Exception:
            continue


def _write_nuscenes_intersections(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write intersection data to map_writer.
    """
    road_blocks = nusc_map.road_block
    for block in road_blocks:
        token = block["token"]
        try:
            if "polygon_token" in block:
                polygon = nusc_map.extract_polygon(block["polygon_token"])
            else:
                continue
            if not polygon.is_valid:
                continue
        except Exception:
            continue

        # Lane group IDs are not directly available; use empty list
        lane_group_ids = []

        map_writer.write_intersection(
            CacheIntersection(
                object_id=token,
                lane_group_ids=lane_group_ids,
                geometry=polygon,
            )
        )


def _write_nuscenes_crosswalks(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write crosswalk data to map_writer.
    """
    ped_crossings = nusc_map.ped_crossing
    for crossing in ped_crossings:
        token = crossing["token"]
        try:
            if "polygon_token" in crossing:
                polygon = nusc_map.extract_polygon(crossing["polygon_token"])
            else:
                continue
            if not polygon.is_valid:
                continue
        except Exception:
            continue

        map_writer.write_crosswalk(
            CacheCrosswalk(
                object_id=token,
                geometry=polygon,
            )
        )


def _write_nuscenes_walkways(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write walkway data to map_writer.
    """
    walkways = nusc_map.walkway
    for walkway in walkways:
        token = walkway["token"]
        try:
            if "polygon_token" in walkway:
                polygon = nusc_map.extract_polygon(walkway["polygon_token"])
            else:
                continue
            if not polygon.is_valid:
                continue
        except Exception:
            continue

        map_writer.write_walkway(
            CacheWalkway(
                object_id=token,
                geometry=polygon,
            )
        )


def _write_nuscenes_carparks(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write carpark data to map_writer.
    """
    carpark_areas = nusc_map.carpark_area
    for carpark in carpark_areas:
        token = carpark["token"]
        try:
            if "polygon_token" in carpark:
                polygon = nusc_map.extract_polygon(carpark["polygon_token"])
            else:
                continue
            if not polygon.is_valid:
                continue
        except Exception:
            continue

        map_writer.write_carpark(
            CacheCarpark(
                object_id=token,
                geometry=polygon,
            )
        )


def _write_nuscenes_generic_drivables(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write generic drivable areas to map_writer.
    """
    # Combine road segments, lanes, and drivable areas
    all_drivables = []

    # Add road segments
    for segment in nusc_map.road_segment:
        try:
            if "polygon_token" in segment:
                polygon = nusc_map.extract_polygon(segment["polygon_token"])
                if polygon.is_valid:
                    all_drivables.append((f"road_segment_{segment['token']}", polygon))
        except Exception:
            continue

    # Add lanes
    for lane in nusc_map.lane:
        try:
            if "polygon_token" in lane:
                polygon = nusc_map.extract_polygon(lane["polygon_token"])
                if polygon.is_valid:
                    all_drivables.append((f"lane_{lane['token']}", polygon))
        except Exception:
            continue

    # Add drivable areas
    for road in nusc_map.drivable_area:
        try:
            if "polygon_token" in road:
                polygon = nusc_map.extract_polygon(road["polygon_token"])
                if polygon.is_valid:
                    all_drivables.append((f"road_{road['token']}", polygon))
        except Exception:
            continue

    for obj_id, geometry in all_drivables:
        map_writer.write_generic_drivable(
            CacheGenericDrivable(
                object_id=obj_id,
                geometry=geometry,
            )
        )


def _write_nuscenes_stop_lines(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write stop line data to map_writer.
    """
    stop_lines = nusc_map.stop_line
    for stop_line in stop_lines:
        token = stop_line["token"]
        try:
            if "polygon_token" in stop_line:
                polygon = nusc_map.extract_polygon(stop_line["polygon_token"])
            else:
                continue
            if not polygon.is_valid:
                continue
        except Exception:
            continue

        # Note: Stop lines are written as generic drivable for compatibility
        map_writer.write_generic_drivable(
            CacheGenericDrivable(
                object_id=token,
                geometry=polygon,
            )
        )


def _write_nuscenes_road_lines(nusc_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """
    Write road line data (dividers) to map_writer.
    """
    # Process road dividers
    road_dividers = nusc_map.road_divider
    for divider in road_dividers:
        token = divider["token"]
        try:
            line = nusc_map.extract_line(divider["line_token"])
            if not line.is_valid:
                continue
        except Exception:
            continue

        # Determine line type
        line_type = _get_road_line_type(divider["line_token"], nusc_map)

        map_writer.write_road_line(
            CacheRoadLine(
                object_id=token,
                road_line_type=line_type,
                polyline=Polyline3D(LineString(line.coords)),
            )
        )

    # Process lane dividers
    lane_dividers = nusc_map.lane_divider
    for divider in lane_dividers:
        token = divider["token"]
        try:
            line = nusc_map.extract_line(divider["line_token"])
            if not line.is_valid:
                continue
        except Exception:
            continue

        line_type = _get_road_line_type(divider["line_token"], nusc_map)

        map_writer.write_road_line(
            CacheRoadLine(
                object_id=token,
                road_line_type=line_type,
                polyline=Polyline3D(LineString(line.coords)),
            )
        )


def _get_lane_boundary(lane_token: str, side: str, nusc_map: NuScenesMap) -> Optional[LineString]:
    """
    Extract lane boundary geometry for a given side.
    """
    lane_record = next((lr for lr in nusc_map.lane if lr["token"] == lane_token), None)
    if not lane_record:
        return None

    divider_segment_nodes_key = f"{side}_lane_divider_segment_nodes"
    if divider_segment_nodes_key in lane_record and lane_record[divider_segment_nodes_key]:
        nodes = lane_record[divider_segment_nodes_key]
        boundary = LineString([(node['x'], node['y']) for node in nodes])
        return boundary

    return None


def _get_lane_group_boundary(segment_token: str, side: str, nusc_map: NuScenesMap) -> Optional[LineString]:
    """
    Extract lane group boundary geometry (simplified).
    """
    # This is a simplified implementation; in practice, may need more robust geometry extraction
    boundary_type = "road_divider" if side == "left" else "lane_divider"

    # Find the segment geometry
    segment = next((rs for rs in nusc_map.road_segment if rs["token"] == segment_token), None)
    if not segment:
        return None

    try:
        segment_geom = nusc_map.extract_polygon(segment["polygon_token"])
    except Exception:
        return None

    # Find nearest boundary of the specified type within a threshold
    nearest = None
    min_dist = float('inf')

    if boundary_type == "road_divider":
        records = nusc_map.road_divider
    else:
        records = nusc_map.lane_divider

    for record in records:
        try:
            line = nusc_map.extract_line(record["line_token"])
            dist = segment_geom.distance(line)
            if dist < 10.0 and dist < min_dist:
                min_dist = dist
                nearest = line
        except Exception:
            continue

    return nearest


def _get_connected_segments(segment_token: str, nusc_map: NuScenesMap):
    """
    Get incoming and outgoing segment connections.
    """
    incoming, outgoing = [], []

    for connector in nusc_map.lane_connector:
        if connector.get("outgoing_lane") == segment_token:
            incoming.append(connector.get("incoming_lane"))
        elif connector.get("incoming_lane") == segment_token:
            outgoing.append(connector.get("outgoing_lane"))

    incoming = [id for id in incoming if id is not None]
    outgoing = [id for id in outgoing if id is not None]

    return incoming, outgoing


def _get_road_line_type(line_token: str, nusc_map: NuScenesMap) -> RoadLineType:
    """
    Map nuscenes line type to RoadLineType.
    """
    nuscenes_to_road_line_type = {
        "SINGLE_SOLID_WHITE": RoadLineType.SOLID_WHITE,
        "DOUBLE_DASHED_WHITE": RoadLineType.DOUBLE_DASH_WHITE,
        "SINGLE_SOLID_YELLOW": RoadLineType.SOLID_YELLOW,
    }

    line_token_to_type = {}
    for lane_record in nusc_map.lane:
        for seg in lane_record.get("left_lane_divider_segments", []):
            token = seg.get("line_token")
            seg_type = seg.get("segment_type")
            if token and seg_type:
                line_token_to_type[token] = seg_type

        for seg in lane_record.get("right_lane_divider_segments", []):
            token = seg.get("line_token")
            seg_type = seg.get("segment_type")
            if token and seg_type:
                line_token_to_type[token] = seg_type

    nuscenes_type = line_token_to_type.get(line_token, "UNKNOWN")
    return nuscenes_to_road_line_type.get(nuscenes_type, RoadLineType.UNKNOWN)


def flip_linestring(linestring: LineString) -> LineString:
    """
    Flip the direction of a LineString.
    """
    return LineString(linestring.coords[::-1])


def lines_same_direction(centerline: LineString, boundary: LineString) -> bool:
    """
    Check if centerline and boundary have the same direction.
    """
    center_start = np.array(centerline.coords[0])
    center_end = np.array(centerline.coords[-1])
    boundary_start = np.array(boundary.coords[0])
    boundary_end = np.array(boundary.coords[-1])

    same_dir_dist = np.linalg.norm(center_start - boundary_start) + np.linalg.norm(center_end - boundary_end)
    opposite_dir_dist = np.linalg.norm(center_start - boundary_end) + np.linalg.norm(center_end - boundary_start)

    return same_dir_dist <= opposite_dir_dist


def align_boundary_direction(centerline: LineString, boundary: LineString) -> LineString:
    """
    Align boundary direction with centerline.
    """
    if not lines_same_direction(centerline, boundary):
        return flip_linestring(boundary)
    return boundary