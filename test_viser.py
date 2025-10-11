import os

from d123.common.multithreading.worker_sequential import Sequential
from d123.common.visualization.viser.viser_viewer import ViserViewer
from d123.datatypes.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from d123.datatypes.scene.scene_filter import SceneFilter
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType

if __name__ == "__main__":

    splits = ["nuplan_mini_test", "nuplan_mini_train", "nuplan_mini_val"]
    # splits = ["nuplan_private_test"]
    # splits = ["carla"]
    splits = ["wopd_val"]
    # splits = ["av2-sensor-mini_train"]
    log_names = None
    scene_uuids = None

    scene_filter = SceneFilter(
        split_names=splits,
        log_names=log_names,
        scene_uuids=scene_uuids,
        duration_s=None,
        history_s=0.0,
        timestamp_threshold_s=None,
        shuffle=True,
        camera_types=[PinholeCameraType.CAM_F0],
    )
    scene_builder = ArrowSceneBuilder(os.environ["D123_DATA_ROOT"])
    worker = Sequential()
    scenes = scene_builder.get_scenes(scene_filter, worker)
    print(f"Found {len(scenes)} scenes")
    visualization_server = ViserViewer(scenes, scene_index=0)
