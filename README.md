<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://kesai.eu/py123d/_static/123D_logo_transparent_white.svg" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://kesai.eu/py123d/_static/123D_logo_transparent_black.svg" width="500">
    <img alt="Logo" src="https://kesai.eu/py123d/_static/123D_logo_transparent_black.svg" width="500">
  </picture>
  <h2 align="center">123D: Unifying Multi-Modal Autonomous Driving Data at Scale</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2605.08084">Paper</a> | <a href="https://youtu.be/Q4q29fpXnx8">Video</a> | <a href="https://kesai.eu/py123d/">Documentation</a></h3>
</h1>

<p align="center">
  <a href="https://pypi.org/project/py123d/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/py123d?color=blue"></a>
  <a href="https://pypi.org/project/py123d/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/py123d"></a>
  <a href="https://github.com/kesai-labs/py123d/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green"></a>
</p>

**One library for autonomous driving datasets.** 123D converts raw data from Argoverse 2, nuScenes, nuPlan, KITTI-360, PandaSet, and Waymo into a unified [Apache Arrow](https://arrow.apache.org/) format, and then gives you a single API to read cameras, lidar, HD maps, and labels across all of them.

## 📰 News

- **2026-05-28**: Released **[NAV123D](https://github.com/DanielDauner/nav123d)**, a reimplementation of NAVSIM on top of 123D.
- **2026-05-08**: Our paper *123D: Unifying Multi-Modal Autonomous Driving Data at Scale* is out on [arXiv](https://arxiv.org/abs/2605.08084).

## ✨ Features

- **Dataset download**: Fetch supported datasets from their official sources via the CLI, and optionally convert directly into the unified format.
- **Hydra-based conversion CLI**: YAML configs to manage your data pipelines.
- **Apache Arrow storage**: columnar, memory-mapped, zero-copy reads. Fast and memory efficient.
- **Multiple sensor codecs**: MP4/JPEG/PNG for cameras; LAZ/Draco/Arrow IPC for lidar.
- **No sensor duplication**: Converted logs reference source camera/lidar files via relative paths, so you don't store sensors twice.
- **Unified API**: Read cameras, lidar, maps, and labels through a single interface, regardless of the source dataset.
- **Built-in visualization**: interactive 3D viewer ([Viser](https://viser.studio/main/)), and [matplotlib](https://matplotlib.org/) plotting.

## 📦 Installation

```bash
pip install py123d
```

Per-dataset extras (e.g. `py123d[av2]`, `py123d[nuscenes]`, `py123d[waymo]`) install the parser dependencies for each dataset on demand. See the [Demo](#demo) below for an example.

## 🚀 Demo

Demo using the Argoverse 2 Sensor dataset, which is publicly readable from S3 and requires no cloud authentication.

The `av2-sensor-stream` config downloads the requested logs/maps into a managed temp directory, converts them into our self-contained Arrow format, and cleans up the source files afterwards. `PY123D_DATA_ROOT` controls where the converted logs/maps are written. The script below installs the Av2 extra, converts the first 3 validation logs (~250 MB each), and launches the Viser viewer:

```bash
# 1. Install
pip install py123d[av2]

export PY123D_DATA_ROOT=/path/to/py123d_data

# 2. Download + Convert
py123d-conversion dataset=av2-sensor-stream \
  dataset.parser.splits='[av2-sensor_val]' \
  dataset.parser.downloader.num_logs=3

# 3. Launch Viewer
py123d-viser scene_filter=av2-sensor
```

Open `http://localhost:8080` to browse the converted scenes interactively.

## 🖼️ Viewer

<p align="center">
  <img src="assets/viser.png" alt="Viser 3D Viewer" width="800">
</p>

## 📊 Supported Datasets

<table>
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th colspan="3" align="center">Scale</th>
      <th colspan="2" align="center">Sensors [#/Hz]</th>
      <th colspan="3" align="center">Annotations [✓/Hz]</th>
    </tr>
    <tr>
      <th></th>
      <th align="left">Dataset</th>
      <th>Year</th>
      <th>Dur. [h]</th>
      <th>Dist. [km]</th>
      <th>Logs [#]</th>
      <th>Cam.</th>
      <th>LiDAR</th>
      <th>3D Box</th>
      <th>Tls.</th>
      <th>Map</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><b>Manual</b></td>
      <td align="left"><a href="https://www.nuscenes.org/">nuScenes</a></td>
      <td>2020</td><td>5.6</td><td>100.9</td><td>1,000</td>
      <td>6&nbsp;/&nbsp;12</td><td>1&nbsp;/&nbsp;20</td>
      <td>✓&nbsp;/&nbsp;2</td><td>✗</td><td>✓</td>
    </tr>
    <tr>
      <td align="left"><a href="https://waymo.com/open/">WOD-Perc.</a></td>
      <td>2020</td><td>6.4</td><td>154.0</td><td>1,150</td>
      <td>5&nbsp;/&nbsp;10</td><td>5&nbsp;/&nbsp;10</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✗</td><td>✓</td>
    </tr>
    <tr>
      <td align="left"><a href="https://www.argoverse.org/">AV2-Sens.</a></td>
      <td>2021</td><td>4.4</td><td>87.5</td><td>1,000</td>
      <td>9&nbsp;/&nbsp;20</td><td>2&nbsp;/&nbsp;10</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✗</td><td>✓</td>
    </tr>
    <tr>
      <td align="left"><a href="https://pandaset.org/">PandaSet</a></td>
      <td>2021</td><td>0.2</td><td>8.3</td><td>103</td>
      <td>6&nbsp;/&nbsp;10</td><td>2&nbsp;/&nbsp;10</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✗</td><td>✗</td>
    </tr>
    <tr>
      <td align="left"><a href="https://www.cvlibs.net/datasets/kitti-360/">KITTI-360</a></td>
      <td>2022</td><td>2.7</td><td>73.7</td><td>9</td>
      <td>4&nbsp;/&nbsp;10</td><td>1&nbsp;/&nbsp;10</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✗</td><td>✓</td>
    </tr>
    <tr>
      <td rowspan="5"><b>Auto-labeled</b></td>
      <td align="left"><a href="https://waymo.com/open/">WOD-Mot.</a></td>
      <td>2021</td><td>574.1</td><td>10,323.5*</td><td>103,354</td>
      <td>✗</td><td>✗</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✓&nbsp;/&nbsp;10</td><td>✓</td>
    </tr>
    <tr>
      <td align="left"><a href="https://www.nuscenes.org/nuplan">nuPlan</a></td>
      <td>2024</td><td>1,174.3</td><td>17,808.6</td><td>15,910</td>
      <td>8&nbsp;/&nbsp;10†</td><td>5&nbsp;/&nbsp;20†</td>
      <td>✓&nbsp;/&nbsp;20</td><td>✓&nbsp;/&nbsp;20</td><td>✓</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;– <a href="https://www.nuscenes.org/nuplan">nuPlan-mini</a></td>
      <td>2024</td><td>7.2</td><td>103.0</td><td>64</td>
      <td>8&nbsp;/&nbsp;10†</td><td>5&nbsp;/&nbsp;20†</td>
      <td>✓&nbsp;/&nbsp;20</td><td>✓&nbsp;/&nbsp;20</td><td>✓</td>
    </tr>
    <tr>
      <td align="left"><a href="https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles">PAI-AV</a></td>
      <td>2025</td><td>1,707.0</td><td>69,265.7</td><td>307,332</td>
      <td>7&nbsp;/&nbsp;30</td><td>1&nbsp;/&nbsp;10</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✗</td><td>✗</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;– <a href="https://github.com/NVIDIA/ncore">NCore</a></td>
      <td>2026</td><td>6.3</td><td>167.6</td><td>1,147</td>
      <td>7&nbsp;/&nbsp;30</td><td>1&nbsp;/&nbsp;10</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✗</td><td>✗</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Synth.</b></td>
      <td align="left"><a href="https://carla.org/">CARLA</a></td>
      <td>2017</td><td><i>var.</i></td><td><i>var.</i></td><td><i>var.</i></td>
      <td><i>var.</i></td><td><i>var.</i></td>
      <td>✓</td><td>✓</td><td>✓</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;– <a href="https://github.com/kesai-labs/lead">L3AD</a></td>
      <td>2026</td><td>7.3</td><td>138.7</td><td>789</td>
      <td>6&nbsp;/&nbsp;10</td><td>2&nbsp;/&nbsp;10</td>
      <td>✓&nbsp;/&nbsp;10</td><td>✓&nbsp;/&nbsp;10</td><td>✓</td>
    </tr>
  </tbody>
</table>

<sub><i>* Computed only from the non-overlapping 20&nbsp;s training files. &nbsp;
† Released for a 120&nbsp;h subset; full coverage on mini.</i></sub>


## 📝 Changelog

<details open>
<summary><b>v0.5.0</b> (2026-05-28)</summary>

- New `Polyline2D` / `PolylineSE2` / `PolylineSE3.subline()` to extract a sub-polyline between two distances along a path.
- New `MapAPI.get_layer_graph()` returning a `networkx.DiGraph` of lane / lane-group predecessor–successor topology.
- Added `BoxDetectionsSE3.box_detections_se2` for projecting SE3 detections to SE2.
- Global runtime settings consolidated into `common.runtime`, configured via Hydra, with configurable LRU cache sizes (e.g. `max_lru_cached_tables`).
- Added a minimal PyTorch dataloader example (`examples/01_torch_dataloader.py`).
- Fixes: nuPlan map predecessor/successor naming for lanes and lane connectors; `np.nan` lane speed limits; scene-sampling stride with `target_iteration_stride`; matplotlib observation rendering.

Breaking changes: `OccupancyMap2D.contains_vectorized` renamed to `contains_points_2d` (now shape-preserving); `dataset_paths` moved from `py123d.common` to `py123d.common.runtime`.

</details>

<details>
<summary><b>v0.4.0</b> (2026-05-19)</summary>

- Added L3AD dataset support (CARLA-derived) with a HuggingFace-hosted downloader that fetches pre-converted Arrow logs directly into `$PY123D_DATA_ROOT`.
- New `SceneAPI.get_modality_between_timestamps()` for time-windowed retrieval.
- nuPlan route and lidar tokens exposed through the custom-modality interface.
- Visualization: expanded viser color and rendering options, with configurable output resolution and frame rate.
- Parser fixes for PAI-AV and ncore: bounding box alignment and file-derived vehicle dimensions.

No breaking changes to the public API, Arrow schema, or CLI entry points.

</details>

<details>
<summary><b>v0.3.0</b> (2026-04-28)</summary>

- Refactored dataset download interface, with new download/stream options for nuScenes, PandaSet (HF mirror), AV2-sensor, WOD-perception, WOMD ([#126](https://github.com/kesai-labs/py123d/pull/126)), and nuPlan.
- Added ncore dataset support with parser, downloader, and on-the-fly conversion ([#125](https://github.com/kesai-labs/py123d/pull/125)).
- Waymo Open Motion: remaining WOMD splits, WOMD-specific fields via custom modalities, and skip-logs / skip-map options for storage-constrained runs.
- Map improvements: speed bumps as surface map objects in Waymo ([#130](https://github.com/kesai-labs/py123d/pull/130)); `align_road_edges_to_traffic` for traffic-aligned road edges ([#123](https://github.com/kesai-labs/py123d/pull/123)).
- Parser fixes across PandaSet (lidar/ego poses, extrinsics), nuPlan ([#128](https://github.com/kesai-labs/py123d/pull/128)), KITTI-360 labels, nuScenes map path in stream mode, WOD motion streaming, and `pai-av` / `ncore` labels.
- Runtime and packaging: corrected sync-table entries when inferring dynamics; Ray executor compatibility; `google-cloud-storage` moved to the Waymo extra.

Includes all fixes from v0.2.1 and v0.2.2. No breaking changes to the public API, Arrow schema, or CLI entry points.

</details>

<details>
<summary><b>v0.2.0</b> (2026-04-14)</summary>

- Transferred repository to [KE:SAI]( https://kesai.eu).

- Aligned ego and agent dynamics to a unified global/ego-frame convention, with velocity/acceleration inference in `LogWriter` from poses for `EgoState` and `BoxDetectionsSE3` ([#119](https://github.com/kesai-labs/py123d/pull/119), [#120](https://github.com/kesai-labs/py123d/pull/120)).
- Improved OpenDRIVE maps: 3D road-edge lifting, lane-boundary reconstruction, and cleaner map-metadata location handling ([#121](https://github.com/kesai-labs/py123d/pull/121)).
- Parser and visualization fixes: NuScenes interpolated parser defaults to 10 Hz sync with camera-pose interpolation; PandaSet extrinsic/undistortion fixes; viser fixes ([#117](https://github.com/kesai-labs/py123d/pull/117)); new matplotlib camera-rig and lidar-reprojection utilities.

No breaking changes to the public API, Arrow schema, or CLI entry points.

</details>

<details>
<summary><b>v0.1.0</b> (2026-03-22)</summary>

- Asynchronous (native-rate) data storage: modalities are now written at their original capture rate, not just at the a frame-wise rate.
- New parser architecture with `BaseLogParser.iter_modalities_async` for native-rate iteration alongside the existing synchronized path.
- Added NVIDIA Physical AI AV dataset support (experimental).
- Added standalone OpenDRIVE / CARLA map parser.
- Refactored `conversion/` module into `parser/` with consistent naming across all dataset parsers.
- Refactored Viser 3D viewer. Adds more control and dark mode.
- Added `LaneType`, `IntersectionType`, `StopZoneType` to map data structure.
- Replaced Waymo heavy dependencies with lightweight protobufs.
- Various fixes to camera-to-global transforms across all datasets.

</details>

<details>
<summary><b>v0.0.9</b> (2026-02-09)</summary>

- Added Waymo Open Motion Dataset support.
- Replaced gpkg map implementation with Arrow-based format for improved performance.
- Added sensor names and timestamps to camera and Lidar data across all datasets.
- Added ego-to-camera transforms in static metadata.
- Implemented geometry builders for PoseSE2/PoseSE3 from arbitrary rotation/translation representations.
- Added support for loading merged point clouds in API.
- Improved map querying speed and OpenDrive lane connectivity handling.
- Added recommended conversion options to dataset YAML configuration files.
- Fixed PandaSet static extrinsics and KITTI-360 timestamp handling.
- Fixed memory issues when converting large datasets (e.g., nuPlan).

</details>

<details>
<summary><b>v0.0.8</b> (2025-11-21)</summary>

- Release of package and documentation.
- Demo data for tutorials.

</details>

## 📚 Citation

```bibtex
@article{Dauner2026ARXIV,
  title={123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  author={Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2605.08084},
  year={2026}
}
```


## ⚖️ License

123D is released under the [Apache License 2.0](LICENSE).
