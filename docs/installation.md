# Installation

## Pip-Install
You can simply install `py123d` for Python versions >=3.9 via PyPI with
```bash
pip install py123d
```
or as editable pip package with
```bash
git clone git@github.com:autonomousvision/py123d.git
cd py123d
pip install -e .
```


## File Structure & Storage
The 123D library converts driving datasets to a unified format. By default, all data is stored in directory of the environment variable `$PY123D_DATA_ROOT`.
```bash
export PY123D_DATA_ROOT="$HOME/py123d_workspace/data"
```
which can be added to your `~/.bashrc` or to your bash scripts. Optionally, you can adjust all dataset paths in the hydra config: `py123d/script/config/common/default_dataset_paths.yaml`.

The 123D conversion includes:
- **Logs:** The logs store continuous driving recordings in a single file, including modalities such as timestamps, ego states, bounding boxes, and sensor references. Logs are stored as `.arrow` files.
- **Maps:** The maps are static and store our unified HD-Map API. Maps can either be defined per-log (e.g. in AV2, Waymo) or globally for a certain location (e.g. nuPlan, nuScenes, CARLA). In the current implementation, we store maps as `.gpkg` files.
- **Sensors:** There are multiple options to store sensor data. Cameras and LiDAR point clouds can either (1) be read from the original dataset or (2) stored within the log file. For cameras, we also support (3) compression with MP4 files, which are written into the `/sensors` directory.

For example, when converting `nuplan-mini` with MP4 compression, the file structure should look the following:
```
$PY123D_DATA_ROOT
├── logs
│   ├── nuplan-mini_test
│   │   ├── 2021.05.25.14.16.10_veh-35_01690_02183.arrow
│   │   ├── ...
│   │   └── 2021.10.06.07.26.10_veh-52_00006_00398.arrow
│   ├── nuplan-mini_train
│   │   └── ...
│   ├── nuplan-mini_train
│   │   └── ...
│   └── ...
├── maps
│   ├── nuplan
│   │   ├── nuplan_sg-one-north.gpkg
│   │   ├── ...
│   │   └── nuplan_us-pa-pittsburgh-hazelwood.gpkg
│   └── ...
└── sensors
    ├── nuplan-mini_test
    │   ├── 2021.05.25.14.16.10_veh-35_01690_02183
    │   │   ├── pcam_b0.mp4
    │   │   ├── ...
    │   │   └── pcam_r2.mp4
    │   └── ...
    └── ...
```
