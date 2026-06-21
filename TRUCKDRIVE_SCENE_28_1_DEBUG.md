# TruckDrive `scene_28_1` Debug Notes (py123d)

This document captures what we learned while debugging odd visualization behavior for TruckDrive `scene_28_1`.

It is intended as a reproducible runbook for colleagues to validate data, transforms, and viewer behavior.

---

## TL;DR

- The scene data and transform chain are valid.
- The early "blank viewer" issue was primarily viewer/runtime setup + UI crash (duplicate lidar dropdown values), not bad scene transforms.
- The "ego path flies off" perception is mostly a global-frame visualization effect (long trajectory + static camera), not a broken SE3 chain.

---

## Minimal Repro Instructions (One Scene)

Use Python 3.13 (recommended; Hydra CLI has issues on 3.14 in this repo stack).

```bash
cd py123d # repo checkout root
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

Set paths:

```bash
export TRUCKDRIVE_DATA_ROOT="pick a path"
export PY123D_DATA_ROOT=${TRUCKDRIVE_DATA_ROOT}/py123d_out
```

Download one scene (all required modalities for full viz):

```bash
py123d-download dataset=truckdrive \
  'dataset.downloader.scenes=[scene_28_1]' \
  'dataset.downloader.modalities=[camera,lidar,poses,calibrations,annotations]'
```

Convert one scene:

```bash
py123d-conversion dataset=truckdrive \
  'dataset.parser.scene_names=[scene_28_1]'
```

Launch viewer for only this scene:

```bash
py123d-viser scene_filter=truckdrive \
  'scene_filter.log_names=[scene_28_1]'
```

Open `http://localhost:8080`.

---

## Quick Validation Checks

Raw + converted data existence:

```bash
ls "$TRUCKDRIVE_DATA_ROOT/scene_28_1/lidar/aeva/joint_lidars/points" | head
ls "$PY123D_DATA_ROOT/logs/truckdrive_val/scene_28_1/sync.arrow"
```

Expected converted files include:

- `sync.arrow`
- `map.arrow`
- `ego_state_se3.arrow`
- `lidar.lidar_merged.arrow`
- `lidar.lidar_front.arrow`
- `lidar.lidar_side_left.arrow`
- `lidar.lidar_side_right.arrow`
- `camera.*.arrow`
- `box_detections_se3.arrow`

---

## Symptoms Observed During Debugging

1. **Blank Viser viewport** with websocket connect/disconnect.
2. Browser console error:
  - `[@mantine/core] Duplicate options are not supported. Option with value "LIDAR_MERGED" was provided more than once`
3. Follow-on JS error:
  - `Cannot read properties of null (reading 'height')`
4. Visual impression that ego trajectory/map "flies off into space".

---

## Root Causes Identified

### 1) Viewer startup/setup pitfalls (not transform math)

- Running `py123d-viser` without first converting to Arrow can produce empty results.
- Missing env vars can break sensor path loading:
  - `TRUCKDRIVE_DATA_ROOT` must be set at viewer runtime because lidar/camera are path-backed.
  - `PY123D_DATA_ROOT` must point at converted Arrow logs.

### 2) UI crash from duplicate lidar dropdown values

- Scene API returned `LIDAR_MERGED` twice for this dataset/view combination.
- Mantine dropdown rejects duplicate option values; React tree error caused viewer instability/blank state.
- Fix: dedupe merged lidar ID/name in `available_lidar_ids`/`available_lidar_names`.

### 3) "Path flies off" perception from global-frame visualization

- This scene covers a long trajectory; camera is not live ego-follow by default.
- In global coordinates, this can look extreme even if math is correct.

---

## Transform Chain Validation Performed

### A) Static + dynamic matrix cross-check vs annotation matrices

For multiple frames, we compared our computed:

- `T_vehicle_to_global = T_aeva_to_global * inverse(T_aeva_to_vehicle)`
- `T_velodyne_to_global = T_vehicle_to_global * T_velodyne_to_vehicle`

against annotation `velodyne2global` matrices in bounding box JSON files.

Observed error was near machine precision:

- translation error ~ `1e-14`
- rotation matrix error ~ `1e-16`

This strongly indicates the SE3 composition direction is correct.

### B) Trajectory orientation sanity

From `poses/gt_trajectory.txt`:

- `z` delta: about `+54.85 m`
- roll range: about `[-0.34 deg, +3.25 deg]`
- pitch range: about `[-3.22 deg, +0.52 deg]`

Interpretation: large elevation gain over long distance, but relatively small roll/pitch; cloud need not show dramatic tilt.

### C) Runtime modality sanity

At representative frames:

- frame 0: merged lidar `(121461, 3)`, boxes `0`
- frame 50: merged lidar `(360426, 3)`, boxes `84`
- total frames: `260`

This indicates data is loading and synchronized.

---

## Non-Issues / Red Herrings We Hit

- Hydra CLI crash on Python 3.14 (`LazyCompletionHelp`): environment/toolchain issue, unrelated to TruckDrive transforms.
- Raw websocket "version mismatch" from manual script probes (`Client: unknown`): caused by non-viser client test, not browser UI path.

---

## What To Ask Colleagues To Validate

1. Run the minimal commands exactly as above on `scene_28_1`.
2. Confirm the viewer opens and remains connected.
3. Toggle layers:
  - Lidar + Ego + Boxes only
  - Then add map and camera frustums
4. Step to frame ~50 and confirm:
  - boxes wrap objects
  - ego sits within cloud context
  - map/lane geometry is plausible
5. Report any console/runtime errors from browser devtools and terminal.

---

## Known Viewer Limitation

- Live interactive camera does not currently "follow ego" by default.
- Ego-follow camera poses are implemented in render export paths (`3rd Person` / `BEV`), not as a live playback toggle.

