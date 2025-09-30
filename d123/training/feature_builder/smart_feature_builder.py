from enum import IntEnum
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from d123.common.datatypes.detection.detection import BoxDetection, BoxDetectionWrapper
from d123.common.datatypes.detection.detection_types import DetectionType
from d123.common.visualization.color.default import TrafficLightStatus
from d123.datasets.maps.abstract_map import MapLayer
from d123.datasets.maps.abstract_map_objects import (
    AbstractCarpark,
    AbstractCrosswalk,
    AbstractGenericDrivable,
    AbstractLaneGroup,
)
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.geometry import BoundingBoxSE2, PolylineSE2, StateSE2
from d123.geometry.geometry_index import StateSE2Index
from d123.geometry.transform.transform_se2 import convert_absolute_to_relative_se2_array

# TODO: Hind feature builder behind abstraction.


class SMARTMapTokenType(IntEnum):
    LANE = 0
    LANE_GROUP_BOUNDARY = 1
    CROSSWALK = 2
    CARPARK = 3
    GENERIC_DRIVABLE = 4


class SMARTMapTokenPlType(IntEnum):
    BOUNDARY = 0
    CENTERLINE = 1
    POLYGON = 2


START_ITERATION: Final[int] = 0


class SMARTFeatureBuilder:
    def __init__(self):
        pass

    def build_features(self, scene: AbstractScene):

        feature_dict = {"scenario_id": scene.token}

        # Optionally, you use a different origin instead
        origin: StateSE2 = scene.get_ego_state_at_iteration(START_ITERATION).bounding_box.center.state_se2

        map_features = _build_map_features(scene, origin)
        feature_dict.update(map_features)
        agent_features = _build_agent_features(scene, origin)
        feature_dict.update(agent_features)

        return feature_dict


def _build_map_features(scene: AbstractScene, origin: StateSE2) -> Dict[str, np.ndarray]:

    # TODO: Add to config
    width, height = 200, 200
    num_points = 3
    segment_length = 5.0

    # create map extent polygon
    map_bounding_box = BoundingBoxSE2(origin, height, width)

    map_api = scene.map_api
    map_objects = map_api.query(
        map_bounding_box.shapely_polygon,
        layers=[
            MapLayer.LANE_GROUP,
            MapLayer.CROSSWALK,
            MapLayer.CARPARK,
            MapLayer.GENERIC_DRIVABLE,
        ],
        predicate="intersects",
    )

    # Traffic light data
    traffic_lights = scene.get_traffic_light_detections_at_iteration(START_ITERATION)

    traj_se2: List[npt.NDArray[np.float64]] = []
    types: List[int] = []
    pl_types: List[int] = []
    pl_light_types: List[int] = []

    # 1. Add lane
    for lane_group in map_objects[MapLayer.LANE_GROUP]:
        lane_group: AbstractLaneGroup
        is_intersection = lane_group.intersection is not None

        for boundary in [lane_group.right_boundary.polyline_se2, lane_group.left_boundary.polyline_se2]:
            boundary_traj_se2 = _split_segments(
                boundary,
                num_points=num_points,
                segment_length=segment_length,
                map_bounding_box=map_bounding_box,
            )
            traj_se2.extend(boundary_traj_se2)
            types.extend([int(SMARTMapTokenType.LANE_GROUP_BOUNDARY)] * len(boundary_traj_se2))
            pl_types.extend([int(SMARTMapTokenPlType.BOUNDARY)] * len(boundary_traj_se2))
            pl_light_types.extend([int(TrafficLightStatus.OFF)] * len(boundary_traj_se2))

        for lane in lane_group.lanes:
            lane_traffic_light = traffic_lights.get_detection_by_lane_id(lane.id)
            centerline = lane.centerline.polyline_se2
            lane_traj_se2 = _split_segments(
                centerline,
                num_points=num_points,
                segment_length=segment_length,
                map_bounding_box=map_bounding_box,
            )

            traj_se2.extend(lane_traj_se2)
            types.extend([int(SMARTMapTokenType.LANE)] * len(lane_traj_se2))
            pl_types.extend([int(SMARTMapTokenPlType.CENTERLINE)] * len(lane_traj_se2))
            if lane_traffic_light is None:
                if is_intersection:
                    pl_light_types.extend([int(TrafficLightStatus.UNKNOWN)] * len(lane_traj_se2))
                else:
                    pl_light_types.extend([int(TrafficLightStatus.OFF)] * len(lane_traj_se2))
            else:
                pl_light_types.extend([int(lane_traffic_light.status)] * len(lane_traj_se2))

    # 2. Crosswalks
    for crosswalk in map_objects[MapLayer.CROSSWALK]:
        crosswalk: AbstractCrosswalk
        crosswalk_traj_se2 = _split_segments(
            crosswalk.outline_3d.polyline_se2,
            num_points=num_points,
            segment_length=segment_length,
            map_bounding_box=map_bounding_box,
        )
        traj_se2.extend(crosswalk_traj_se2)
        types.extend([int(SMARTMapTokenType.CROSSWALK)] * len(crosswalk_traj_se2))
        pl_types.extend([int(SMARTMapTokenPlType.POLYGON)] * len(crosswalk_traj_se2))
        pl_light_types.extend([int(TrafficLightStatus.OFF)] * len(crosswalk_traj_se2))

    # 3. Parking
    for carpark in map_objects[MapLayer.CARPARK]:
        carpark: AbstractCarpark
        carpark_traj_se2 = _split_segments(
            carpark.outline_3d.polyline_se2,
            num_points=num_points,
            segment_length=segment_length,
            map_bounding_box=map_bounding_box,
        )
        traj_se2.extend(carpark_traj_se2)
        types.extend([int(SMARTMapTokenType.CARPARK)] * len(carpark_traj_se2))
        pl_types.extend([int(SMARTMapTokenPlType.POLYGON)] * len(carpark_traj_se2))
        pl_light_types.extend([int(TrafficLightStatus.OFF)] * len(carpark_traj_se2))

    # 4. Generic drivable
    for generic_drivable in map_objects[MapLayer.GENERIC_DRIVABLE]:
        generic_drivable: AbstractGenericDrivable
        drivable_traj_se2 = _split_segments(
            generic_drivable.outline_3d.polyline_se2,
            num_points=num_points,
            segment_length=segment_length,
            map_bounding_box=map_bounding_box,
        )
        traj_se2.extend(drivable_traj_se2)
        types.extend([int(SMARTMapTokenType.GENERIC_DRIVABLE)] * len(drivable_traj_se2))
        pl_types.extend([int(SMARTMapTokenPlType.POLYGON)] * len(drivable_traj_se2))
        pl_light_types.extend([int(TrafficLightStatus.OFF)] * len(drivable_traj_se2))

    assert len(traj_se2) == len(types) == len(pl_types) == len(pl_light_types)

    traj_se2 = np.array(traj_se2, dtype=np.float64)
    types = np.array(types, dtype=np.uint8)
    pl_types = np.array(pl_types, dtype=np.uint8)
    pl_light_types = np.array(pl_light_types, dtype=np.uint8)
    traj_se2 = convert_absolute_to_relative_se2_array(origin, traj_se2)

    return {
        "map_save": {
            "traj_pos": traj_se2[..., StateSE2Index.XY],
            "traj_theta": traj_se2[..., 0, StateSE2Index.YAW],
        },
        "pt_token": {
            "type": types,
            "pl_type": pl_types,
            "light_type": pl_light_types,
            "num_nodes": len(traj_se2),
        },
    }


def _build_agent_features(scene: AbstractScene, origin: StateSE2) -> None:
    iteration_indices = np.arange(
        -scene.get_number_of_history_iterations(),
        scene.get_number_of_iterations(),
    )
    # print(iteration_indices[scene.get_number_of_history_iterations()])
    num_steps = len(iteration_indices)

    target_types = [DetectionType.VEHICLE, DetectionType.PEDESTRIAN, DetectionType.BICYCLE]
    box_detections_list = [scene.get_box_detections_at_iteration(iteration) for iteration in iteration_indices]
    target_detections: List[List[BoxDetection]] = []
    target_indices: List[List[int]] = []
    for box_detections in box_detections_list:
        in_types, _, in_indices = _filter_agents_by_type(box_detections, target_types)
        target_detections.append(in_types)
        target_indices.append(in_indices)

    # initial_agents = [
    #     detection.metadata.track_token for detection in target_detections[scene.get_number_of_history_iterations()]
    # ]
    other_start_iteration = scene.get_number_of_history_iterations()
    initial_agents = [detection.metadata.track_token for detection in target_detections[other_start_iteration]]
    initial_indices = target_indices[other_start_iteration]
    num_agents = len(initial_agents) + 1  # + 1 for ego vehicle

    def detection_type_to_index(detection_type: DetectionType) -> int:
        if detection_type == DetectionType.VEHICLE:
            return 0
        elif detection_type == DetectionType.PEDESTRIAN:
            return 1
        elif detection_type == DetectionType.BICYCLE:
            return 2
        else:
            raise ValueError(f"Unsupported detection type: {detection_type}")

    # Fill role, id, type arrays
    role = np.zeros((num_agents, 3), dtype=bool)
    id = np.zeros((num_agents), dtype=np.int64)
    type = np.zeros((num_agents), dtype=np.uint8)
    extent = np.zeros((num_agents, 3), dtype=np.float32)

    for agent_idx, agent_token in enumerate(initial_agents):
        detection = box_detections_list[other_start_iteration].get_detection_by_track_token(agent_token)
        assert detection is not None, f"Agent {agent_token} not found in initial detections."

        role_idx = 2 if detection.metadata.detection_type == DetectionType.VEHICLE else 1
        role[agent_idx, role_idx] = True
        id[agent_idx] = initial_indices[agent_idx]
        type[agent_idx] = detection_type_to_index(detection.metadata.detection_type)
        extent[agent_idx] = [
            detection.bounding_box.length,
            detection.bounding_box.width,
            1.0,
        ]  # NOTE: fill height with 1.0 as placeholder (not always available

    # Fill ego vehicle data
    role[-1, 0] = True
    id[-1] = -1  # Use -1 for ego vehicle
    type[-1] = detection_type_to_index(DetectionType.VEHICLE)

    # Fill role, id, type arrays
    valid_mask = np.zeros((num_agents, num_steps), dtype=bool)
    position = np.zeros((num_agents, num_steps, 3), dtype=np.float64)
    heading = np.zeros((num_agents, num_steps), dtype=np.float64)
    velocity = np.zeros((num_agents, num_steps, 2), dtype=np.float64)

    for time_idx, iteration in enumerate(iteration_indices):
        for agent_idx, agent_token in enumerate(initial_agents):
            detection = box_detections_list[time_idx].get_detection_by_track_token(agent_token)
            if detection is None:
                continue

            valid_mask[agent_idx, time_idx] = True

            state_se2 = detection.bounding_box.center.state_se2
            local_se2_array = convert_absolute_to_relative_se2_array(origin, state_se2.array)
            position[agent_idx, time_idx, :2] = local_se2_array[..., StateSE2Index.XY]
            # position[agent_idx, time_idx, 2] = ... #  Is this the z dimension?
            heading[agent_idx, time_idx] = local_se2_array[..., StateSE2Index.YAW]
            velocity[agent_idx, time_idx, :] = detection.velocity.array[:2]  # already in local of agent

        # Fill ego vehicle data
        ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
        valid_mask[-1, time_idx] = True
        local_se2_array = convert_absolute_to_relative_se2_array(
            origin, ego_vehicle_state.bounding_box.center.state_se2.array
        )
        position[-1, time_idx, :2] = local_se2_array[..., StateSE2Index.XY]
        # position[-1, time_idx, 2] = ... #  Is this the z dimension?
        heading[-1, time_idx] = local_se2_array[..., StateSE2Index.YAW]
        velocity[-1, time_idx, :] = ego_vehicle_state.dynamic_state_se3.velocity.array[:2]  # already in local of agent

    return {
        "agent": {
            "num_nodes": num_agents,
            "valid_mask": valid_mask,
            "role": role,
            "id": id,
            "type": type,
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": extent,  # Placeholder for shape, if needed
        }
    }


def _split_segments(
    polyline: PolylineSE2,
    num_points: int,
    segment_length: float,
    map_bounding_box: Optional[BoundingBoxSE2] = None,
) -> List[npt.NDArray[np.float64]]:

    segments_distances = np.concatenate([np.arange(0.0, polyline.length, step=segment_length), [polyline.length]])
    polygon = map_bounding_box.shapely_polygon if map_bounding_box is not None else None

    segments = []
    for segment_start, segment_end in zip(segments_distances[:-1], segments_distances[1:]):
        include_endpoint = True
        poses = polyline.interpolate(
            np.linspace(
                segment_start,
                segment_end,
                num=num_points,
                endpoint=include_endpoint,
            )
        )
        if polygon is not None:
            points_shapely = shapely.creation.points(poses[(0, -1), :2])
            in_map = any(polygon.contains(points_shapely))
            if not in_map:
                continue
        segments.append(poses)

    return segments


def _filter_agents_by_type(
    detections: BoxDetectionWrapper, detection_types: List[DetectionType]
) -> Tuple[List[BoxDetection], List[BoxDetection], List[int]]:

    in_types, not_in_types, in_indices = [], [], []
    for detection_idx, detection in enumerate(detections):
        if detection.metadata.detection_type in detection_types:
            in_types.append(detection)
            in_indices.append(detection_idx)
        else:
            not_in_types.append(detection)

    return in_types, not_in_types, in_indices
