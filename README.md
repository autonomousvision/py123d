<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_white.svg" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_black.svg" width="500">
    <img alt="Logo" src="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_black.svg" width="500">
  </picture>
  <h2 align="center">123D: A Library for Driving Datasets</h1>
  <h3 align="center"><a href="https://youtu.be/Q4q29fpXnx8">Video</a> | <a href="https://autonomousvision.github.io/py123d/">Documentation</a>
</h1>


## Features

- Unified API for driving data, including sensor data, maps, and labels.
- Support for multiple sensors storage formats.
- Fast dataformat based on [Apache Arrow](https://arrow.apache.org/).
- Visualization tools with [matplotlib](https://matplotlib.org/) and [Viser](https://viser.studio/main/).


> **Warning**
>
> This library is under active development and not stable. The API and features may change in future releases.
> Please report issues, feature requests, or other feedback by opening an issue.


## Changelog

- **`[2026-02-23]`** v0.0.9
  - Added preliminary Waymo Open Motion Dataset support.
  - Added support for nuScenes interpolated to 10Hz.
  - Replaced gpkg map implementation with Arrow-based format for improved performance.
  - Added sensor names and timestamps to camera and Lidar data across all datasets.
  - Added ego-to-camera transforms in static metadata.
  - Added support for loading merged point clouds in API.
  - Improvements to geometry module, in terms of speed and syntax.
  - Improved map querying speed and OpenDrive lane connectivity handling.
  - Added recommended conversion options to dataset YAML configuration files.
  - Improvements in dataset conversion for sensor sync, speed, and memory requirements.

- **`[2025-11-21]`** v0.0.8 (silent release)
  - Release of package and documentation.
  - Demo data for tutorials.
