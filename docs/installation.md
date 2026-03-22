# Installation

## Pip-Install
You can simply install `py123d` for Python versions >=3.9 via PyPI with
```bash
pip install py123d
```
or as editable pip package with
```bash
mkdir -p $HOME/py123d_workspace; cd $HOME/py123d_workspace # Optional
git clone git@github.com:autonomousvision/py123d.git
cd py123d
pip install -e .
```

## File Structure & Storage
The 123D library converts driving datasets to a unified format. By default, all data is stored in directory of the environment variable `$PY123D_DATA_ROOT`.
For example, you can use.

```bash
export PY123D_DATA_ROOT="$HOME/py123d_workspace/data"
```
which can be added to your `~/.bashrc` or to your bash scripts. Optionally, you can adjust all dataset paths in the hydra config: `py123d/script/config/common/default_dataset_paths.yaml`.

The 123D conversion includes:
- **Logs:** Each log is a directory containing per-modality `.arrow` files (e.g. ego state, cameras, lidar, bounding boxes, traffic lights) and a `sync.arrow` synchronization table. Each modality is stored at its native capture rate. When using MP4 camera compression, the `.mp4` files are also stored in the log directory.
- **Maps:** The maps are static and store our unified HD-Map API. Maps can either be defined per-log (e.g. in AV2, Waymo) or globally for a certain location (e.g. nuPlan, nuScenes, CARLA). We use `.arrow` files to store maps.
- **Sensors:** Camera and lidar data can either be (1) read from the original dataset via relative paths stored in the `.arrow` files or (2) embedded as binary data in the `.arrow` files. Cameras additionally support (3) MP4 compression, stored alongside the `.arrow` files in the log directory.

For example, when converting `nuplan-mini` with MP4 compression and using `PY123D_DATA_ROOT="$HOME/py123d_workspace/data"`, the file structure would look the following way:
```
~/py123d_workspace/
├── data/
│   ├── logs/
│   │   ├── nuplan-mini_test/
│   │   │   ├── 2021.05.25.14.16.10_veh-35_01690_02183/
│   │   │   │   ├── sync.arrow
│   │   │   │   ├── ego_state_se3.arrow
│   │   │   │   ├── box_detections_se3.arrow
│   │   │   │   ├── traffic_light_detections.arrow
│   │   │   │   ├── camera.pcam_f0.arrow
│   │   │   │   ├── ...
│   │   │   │   ├── camera.pcam_b0.arrow
│   │   │   │   ├── camera.pcam_f0.mp4       (optional, if MP4 compression)
│   │   │   │   ├── ...
│   │   │   │   └── lidar.lidar_merged.arrow
│   │   │   └── ...
│   │   ├── nuplan-mini_train/
│   │   │   └── ...
│   │   └── ...
│   └── maps/
│       ├── nuplan/
│       │   ├── nuplan_sg-one-north.arrow
│       │   ├── ...
│       │   └── nuplan_us-pa-pittsburgh-hazelwood.arrow
│       └── ...
└── py123d/ (repository)
    └── ...
```
