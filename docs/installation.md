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
- **Logs:** The logs store continuous driving recordings in a single file, including modalities such as timestamps, ego states, bounding boxes, and sensor references. Logs are stored as `.arrow` files.
- **Maps:** The maps are static and store our unified HD-Map API. Maps can either be defined per-log (e.g. in AV2, Waymo) or globally for a certain location (e.g. nuPlan, nuScenes, CARLA). We also use `.arrow` files to store maps.
- **Sensors:** There are multiple options to store sensor data. Cameras and Lidar point clouds can either (1) be read from the original dataset or (2) stored within the log file. For cameras, we also support (3) compression with MP4 files, which are written into the `/sensors` directory.

For example, when converting `nuplan-mini` with MP4 compression and using `PY123D_DATA_ROOT="$HOME/py123d_workspace/data"`, the file structure would look the following way:
```
~/py123d_workspace/
├── data/
│   ├── logs
│   │   ├── nuplan-mini_test
│   │   │   ├── 2021.05.25.14.16.10_veh-35_01690_02183.arrow
│   │   │   ├── ...
│   │   │   └── 2021.10.06.07.26.10_veh-52_00006_00398.arrow
│   │   ├── nuplan-mini_train
│   │   │   └── ...
│   │   ├── nuplan-mini_train
│   │   │   └── ...
│   │   └── ...
│   ├── maps
│   │   ├── nuplan
│   │   │   ├── nuplan_sg-one-north.arrow
│   │   │   ├── ...
│   │   │   └── nuplan_us-pa-pittsburgh-hazelwood.arrow
│   │   └── ...
│   └── sensors
│       ├── nuplan-mini_test
│       │   ├── 2021.05.25.14.16.10_veh-35_01690_02183
│       │   │   ├── pcam_b0.mp4
│       │   │   ├── ...
│       │   │   └── pcam_r2.mp4
│       │   └── ...
│       └── ...
└── py123d/ (repository)
    └── ...
```

## Demo data


You can test 123D with demo data from [nuPlan](nuplan), [nuScenes](nuscenes), [PandaSet](pandaset), [Argoverse 2 - Sensor](av2_sensor), and [CARLA](carla). Please be aware of the respective licenses, that are included in the download. You can use the following script:

```bash
# Create the data root and a temporary folder.
mkdir -p $PY123D_DATA_ROOT
mkdir -p ./temp

# Download the demo data.
wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/123d/demo_v0.0.8/data.zip

# Unzip, sync, and clean up.
unzip -o data.zip -d ./temp
rsync -av ./temp/data/* $PY123D_DATA_ROOT
rm -r ./temp & rm -r data.zip
```
