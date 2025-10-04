import os

from d123.common.multithreading.worker_sequential import Sequential
from d123.common.visualization.viser.viser_viewer import ViserViewer
from d123.datatypes.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from d123.datatypes.scene.scene_filter import SceneFilter
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCameraType

if __name__ == "__main__":

    splits = ["nuplan_private_test"]
    # splits = ["carla"]
    # splits = ["wopd_train"]
    # splits = ["av2-sensor-mini_train"]
    log_names = None

    scene_tokens = None

    scene_filter = SceneFilter(
        split_names=splits,
        log_names=log_names,
        scene_tokens=scene_tokens,
        duration_s=18,
        history_s=0.5,
        timestamp_threshold_s=10,
        shuffle=True,
        camera_types=[PinholeCameraType.CAM_F0],
    )
    scene_builder = ArrowSceneBuilder(os.environ["D123_DATA_ROOT"])
    worker = Sequential()
    scenes = scene_builder.get_scenes(scene_filter, worker)
    print(f"Found {len(scenes)} scenes")
    visualization_server = ViserViewer(scenes, scene_index=0)
