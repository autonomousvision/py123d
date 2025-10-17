# Installation

Note, the following installation assumes the following folder structure: TODO UPDATE
```
~/py123d_workspace
├── py123d
├── exp
│   └── ...
└── data
    ├── maps
    │   ├── carla_town01.gpkg
    │   ├── carla_town02.gpkg
    │   ├── ...
    │   └── nuplan_us-pa-pittsburgh-hazelwood.gpkg
    ├── nuplan_mini_test
    │   ├── 2021.05.25.14.16.10_veh-35_01690_02183.arrow
    │   ├── 2021.06.03.12.02.06_veh-35_00233_00609.arrow
    │   ├── ...
    │   └── 2021.10.06.07.26.10_veh-52_00006_00398.arrow
    ├── nuplan_mini_train
    │   └── ...
    └── nuplan_mini_test
        └── ...
```


First you need to create a new conda environment and install `py123d` as editable pip package.
```bash
conda create -n py123d python=3.12
conda activate py123d
pip install -e .
```

Next, you need add the following environment variables in your `.bashrc`:
```bash
export PY123D_DATA_ROOT="$HOME/py123d_workspace/data"
```

<!-- # CARLA
export CARLA_SIMULATOR_ROOT="$HOME/carla_workspace/carla_garage/carla"

# nuPlan
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/dataset/maps" -->