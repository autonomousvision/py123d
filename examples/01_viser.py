from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.multithreading.worker_sequential import Sequential
from py123d.visualization.viser.viser_viewer import ViserViewer

if __name__ == "__main__":
    # splits = ["kitti360_train"]
    # splits = ["nuscenes-mini_val", "nuscenes-mini_train"]
    # splits = ["nuplan-mini_test", "nuplan-mini_train", "nuplan-mini_val"]
    # splits = ["nuplan_private_test"]
    # splits = ["carla_test"]
    splits = ["wopd_val"]
    # splits = ["av2-sensor_train"]
    # splits = ["pandaset_test", "pandaset_val", "pandaset_train"]
    # log_names = ["2021.08.24.13.12.55_veh-45_00386_00472"]
    # log_names = ["2013_05_28_drive_0000_sync"]
    # log_names = ["2013_05_28_drive_0000_sync"]
    log_names = None
    splits = None
    # scene_uuids = ["87bf69e4-f2fb-5491-99fa-8b7e89fb697c"]
    scene_uuids = None

    scene_filter = SceneFilter(
        split_names=splits,
        log_names=log_names,
        scene_uuids=scene_uuids,
        duration_s=5.0,
        history_s=0.0,
        timestamp_threshold_s=5.0,
        shuffle=True,
        # pinhole_camera_types=[PinholeCameraType.PCAM_F0],
    )
    scene_builder = ArrowSceneBuilder()
    worker = Sequential()
    scenes = scene_builder.get_scenes(scene_filter, worker)
    print(f"Found {len(scenes)} scenes")
    visualization_server = ViserViewer(scenes, scene_index=0)
