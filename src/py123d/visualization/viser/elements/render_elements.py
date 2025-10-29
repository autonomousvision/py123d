from py123d.conversion.utils.sensor_utils.camera_conventions import convert_camera_convention
from py123d.datatypes.scene.abstract_scene import AbstractScene
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.geometry_index import StateSE3Index
from py123d.geometry.se import StateSE3
from py123d.geometry.transform.transform_se3 import translate_se3_along_body_frame
from py123d.geometry.vector import Vector3D


def get_ego_3rd_person_view_position(
    scene: AbstractScene,
    iteration: int,
    initial_ego_state: EgoStateSE3,
) -> StateSE3:
    scene_center_array = initial_ego_state.center.point_3d.array
    ego_pose = scene.get_ego_state_at_iteration(iteration).rear_axle_se3.array
    ego_pose[StateSE3Index.XYZ] -= scene_center_array
    ego_pose_se3 = StateSE3.from_array(ego_pose)
    ego_pose_se3 = translate_se3_along_body_frame(ego_pose_se3, Vector3D(-10.0, 0.0, 5.0))

    return convert_camera_convention(
        ego_pose_se3,
        from_convention="pXpZmY",
        to_convention="pZmYpX",
    )
