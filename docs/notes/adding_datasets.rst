Adding Datasets
===============

This tutorial walks through adding a new source dataset to the 123D conversion
pipeline. A dataset is integrated by implementing a small set of **parser**
classes that read the raw data and yield it as 123D's unified datatypes; the
framework then handles synchronization, Arrow serialization, and exposing the
result through the unified API.

.. note::
  The fastest way to add a dataset is to **copy the closest existing parser and
  adapt it**, rather than starting from a blank file. Good starting points:

  * A sensor dataset with cameras, lidar, and boxes → :doc:`nuScenes </datasets/nuscenes>`
    (``src/py123d/parser/nuscenes/``) or :doc:`Argoverse 2 </datasets/av2>`
    (``src/py123d/parser/av2/``).
  * A dataset with an automated downloader / streaming conversion →
    :doc:`nuPlan </datasets/nuplan>` (``src/py123d/parser/nuplan/``).

  The sections below explain the *contract* each piece must satisfy, so you know
  what to keep and what to replace.


1. General
----------

Architecture overview
~~~~~~~~~~~~~~~~~~~~~~~

Conversion is organized around three abstract base classes defined in
:class:`~py123d.parser.base_dataset_parser.BaseDatasetParser`. A top-level
*dataset parser* produces lightweight, **picklable** per-log and per-map handles.
The orchestrator constructs these once on the main process, then distributes them
to Ray workers, which perform the heavy I/O lazily and hand the parsed data to the
Arrow writers:

.. code-block:: text

  BaseDatasetParser
    ├── get_log_parsers()  ──▶  [ LogParser_1, ..., LogParser_N ]  ──▶  ArrowLogWriter
    └── get_map_parsers()  ──▶  [ MapParser_1, ..., MapParser_M ]  ──▶  ArrowMapWriter
                                                                            │
                                                                            ▼
                                                                  Arrow storage ──▶ unified API

Each new dataset adds a package under ``src/py123d/parser/<dataset>/``. Existing
packages — ``nuscenes/``, ``nuplan/``, ``wod/``, ``av2/``, ``kitti360/``,
``pandaset/`` — are the canonical references.

The dataset parser
~~~~~~~~~~~~~~~~~~~~

Subclass :class:`~py123d.parser.base_dataset_parser.BaseDatasetParser` and
implement two methods:

.. code-block:: python

  from py123d.parser.base_dataset_parser import BaseDatasetParser

  class MyDatasetParser(BaseDatasetParser):

      def get_log_parsers(self) -> list[BaseLogParser]:
          """One lightweight, picklable handle per log/scene."""
          ...

      def get_map_parsers(self) -> list[BaseMapParser]:
          """One handle per map region (return [] if the dataset has no maps)."""
          ...

.. important::
  The returned handles are pickled and shipped to worker processes. Keep them
  **lightweight** — store only paths and parameters, not open file handles,
  database connections, or decoded sensor data. All expensive I/O belongs inside
  the iterators (covered below), which run on the worker.

Optional and gated dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset-specific third-party packages (devkits, format readers, cloud SDKs) must
**not** be hard dependencies of 123D. Declare them as an extra in
``pyproject.toml`` under ``[project.optional-dependencies]``:

.. code-block:: toml

  [project.optional-dependencies]
  mydataset = ["mydataset-devkit==1.2.0"]

Then gate the import at the top of your parser (or lazily inside the method that
needs it) with :func:`~py123d.common.utils.dependencies.check_dependencies`, which
raises a helpful, install-instruction error if the package is missing:

.. code-block:: python

  from py123d.common.utils.dependencies import check_dependencies

  check_dependencies(["mydataset"], optional_name="mydataset")
  # ImportError: Missing 'mydataset'. Install with: pip install py123d[mydataset]

Entry point
~~~~~~~~~~~~

Conversion is driven by the ``py123d-conversion`` CLI (defined in
``[project.scripts]``, mapping to
:func:`py123d.script.run_conversion.main`). It uses `Hydra
<https://hydra.cc>`_ to instantiate your parser from a YAML config via its
``_target_`` field (see :ref:`section 5 <adding-datasets-hydra>`):

.. code-block:: bash

  py123d-conversion dataset=mydataset


2. Log Parser
-------------

A *log parser* is a picklable handle to a single log/scene. Subclass
:class:`~py123d.parser.base_dataset_parser.BaseLogParser` and implement the log
metadata getter and the synchronized-frame iterator.

Log metadata
~~~~~~~~~~~~~

``get_log_metadata()`` returns a
:class:`~py123d.datatypes.metadata.log_metadata.LogMetadata` describing the log:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``dataset``
     - Dataset name, lowercase (e.g. ``"mydataset"``).
   * - ``split``
     - Data split, prefixed by dataset (e.g. ``"mydataset_train"``).
   * - ``log_name``
     - Unique name of the log.
   * - ``location``
     - Geographic location (optional); links a log to a per-location map.
   * - ``map_metadata``
     - Associated :class:`~py123d.datatypes.metadata.map_metadata.MapMetadata`, if any.
   * - ``version``
     - 123D version that produced the metadata (defaults to the current version).

Iterating modalities
~~~~~~~~~~~~~~~~~~~~~~

``iter_modalities_sync()`` lazily yields one
:class:`~py123d.parser.base_dataset_parser.ModalitiesSync` per synchronized frame.
Each frame bundles a timestamp with a list of 123D *modalities* — the unified
:class:`~py123d.datatypes.modalities.base_modality.BaseModality` types parsed from
the dataset's native formats (ego state, bounding boxes, cameras, lidar, radar,
…):

.. code-block:: python

  from py123d.parser.base_dataset_parser import ModalitiesSync

  def iter_modalities_sync(self):
      for frame in self._read_frames():
          modalities = [
              self._parse_ego_state(frame),
              self._parse_boxes(frame),
              *self._parse_cameras(frame),
              self._parse_lidar(frame),
          ]
          yield ModalitiesSync(timestamp=frame.timestamp, modalities=modalities)

Parsed sensor datatypes and lazy I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sensor data is the bulk of any dataset, so 123D never decodes it during parsing.
Instead, yield the lightweight *parsed* modality helpers, which carry **only a
file path (or raw bytes) plus metadata** and defer decoding until the writer
actually needs it:

* :class:`~py123d.parser.base_dataset_parser.ParsedCamera` — camera metadata, a
  ``camera_to_global_se3`` pose, and either a ``dataset_root``/``relative_path``
  pair **or** an in-memory ``byte_string``.
* :class:`~py123d.parser.base_dataset_parser.ParsedLidar` — lidar metadata,
  ``start_timestamp``/``end_timestamp`` (the sweep window), a
  ``dataset_root``/``relative_path``, and optional ``load_kwargs``.
* :class:`~py123d.parser.base_dataset_parser.ParsedRadar` — radar metadata, a
  single ``timestamp`` (a radar scan is an instantaneous snapshot), a file path,
  and optional ``load_kwargs``.

This path-not-payload design keeps memory flat and conversion fast: a worker can
stream thousands of frames without ever holding a full point cloud or decoded
image in RAM.

.. tip::
  ``load_kwargs`` is a generic, dataset-specific dictionary forwarded verbatim to
  the point-cloud loader and persisted in the Arrow frame under the ``"path"``
  store option. nuScenes uses it to carry the keyframe's panoptic-label path
  alongside the lidar file.

Where to add sensor I/O
~~~~~~~~~~~~~~~~~~~~~~~~~

If your dataset uses a point-cloud or image format not already supported, add a
loader under ``src/py123d/common/io/{camera,lidar,radar}/`` and register it in the
dispatcher
:func:`py123d.common.io.lidar.path_lidar_io.load_point_cloud_data_from_path`
(for lidar). The dispatcher routes by file extension / store option and is where
``load_kwargs`` is consumed.

Sync vs. async parsing
~~~~~~~~~~~~~~~~~~~~~~~~

By default, modalities are written as synchronized frames. For datasets whose
sensors run at different native rates (e.g. cameras at 20 Hz, lidar at 10 Hz), you
can override
:meth:`~py123d.parser.base_dataset_parser.BaseLogParser.iter_modalities_async`,
which yields each modality independently as it is produced. The default
implementation simply unwraps the synchronized frames, so overriding is optional.
Async writing is selected by the ``async_conversion`` flag in the config
(:ref:`section 5 <adding-datasets-hydra>`).


3. Map Parser
-------------

.. warning::
  Maps are usually the trickiest part to unify — HD-map schemas vary widely across
  datasets. Maps are **optional**: if your dataset ships none, return ``[]`` from
  ``get_map_parsers()`` and skip this section.

A *map parser* is a picklable handle to one map region. Subclass
:class:`~py123d.parser.base_dataset_parser.BaseMapParser` and implement:

.. code-block:: python

  from py123d.parser.base_dataset_parser import BaseMapParser

  class MyMapParser(BaseMapParser):

      def get_map_metadata(self) -> MapMetadata:
          ...

      def iter_map_objects(self):
          """Yield map objects lazily, one at a time."""
          yield from self._parse_lanes()
          yield from self._parse_crosswalks()
          # ...

Map metadata
~~~~~~~~~~~~~

``get_map_metadata()`` returns a
:class:`~py123d.datatypes.metadata.map_metadata.MapMetadata`. Two fields deserve
particular attention:

* ``map_has_z`` — whether the map geometry carries elevation (3D) or is 2D only.
* ``map_is_per_log`` — whether each log has its own map (e.g. KITTI-360) or maps
  are shared per location across logs (e.g. nuScenes, nuPlan).

Supported map elements
~~~~~~~~~~~~~~~~~~~~~~~~

``iter_map_objects()`` yields
:class:`~py123d.datatypes.map_objects.base_map_objects.BaseMapObject` instances.
Each object is one of the supported element types (the
:class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` enum), defined in
``src/py123d/datatypes/map_objects/map_objects.py``:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Element
     - Geometry
     - Notes
   * - ``Lane``
     - surface
     - Individual driving lane with boundaries and centerline. Lane connectors are
       also yielded as ``Lane`` objects.
   * - ``LaneGroup``
     - surface
     - Group of lanes travelling in the same direction.
   * - ``Intersection``
     - surface
     - Intersection area.
   * - ``Crosswalk``
     - surface
     - Pedestrian crossing.
   * - ``Walkway``
     - surface
     - Pedestrian walkway.
   * - ``Carpark``
     - surface
     - Parking area.
   * - ``GenericDrivable``
     - surface
     - Other drivable surface.
   * - ``StopZone``
     - surface
     - Zone governed by a stop sign / light.
   * - ``RoadEdge``
     - line
     - Road boundary.
   * - ``RoadLine``
     - line
     - Painted lane marking.
   * - ``SpeedBump``
     - surface
     - Speed-reduction area.

Surface elements derive from ``BaseMapSurfaceObject`` (a polygon ``outline``);
line elements derive from ``BaseMapLineObject`` (a ``polyline``).

.. note::
  Need a map element type that isn't listed above? New layers can be added — please
  open an issue or PR (see :doc:`contributing`) so the schema stays consistent
  across datasets.


4. Downloader (optional)
------------------------

A downloader is **optional**, but it can significantly lower the onboarding
barrier: contributors get the data with a single command, and the same component
powers *streaming* conversion (download a subset to a temp directory, convert from
it, delete it).

Subclass :class:`~py123d.parser.base_downloader.BaseDownloader`, placing it at
``src/py123d/parser/<dataset>/<dataset>_download.py``:

.. code-block:: python

  from py123d.parser.base_downloader import BaseDownloader

  class MyDatasetDownloader(BaseDownloader):

      def download(self) -> None:
          """Fetch the selected data into ``self.output_dir``."""
          ...

The base class provides ``output_dir`` (destination; ``None`` means a temp
directory is assigned at runtime) and ``dry_run`` (log the plan without writing).

Integration follows two paths, both demonstrated by
:class:`~py123d.parser.nuplan.nuplan_download.NuplanDownloader`:

* **Standalone CLI** — ``py123d-download dataset=mydataset`` instantiates the
  downloader and calls ``download()``.
* **Streaming during conversion** — the parser accepts an optional ``downloader=``
  argument, redirects its ``output_dir`` to a session-scoped temp directory,
  downloads before reading, and cleans up on destruction.


.. _adding-datasets-hydra:

5. Hydra config
---------------

Conversion configs live in
``src/py123d/script/config/conversion/dataset/``, one ``<name>.yaml`` per dataset
variant. The base defaults are in ``default_conversion.yaml``. Each dataset config
has four sections:

* ``parser`` — instantiates your ``BaseDatasetParser`` (via ``_target_``) and its
  data-root paths / options.
* ``log_writer_config`` — storage options
  (``camera_store_option``, ``lidar_store_option``, ``lidar_codec``),
  the ``async_conversion`` flag, and inference toggles.
* ``log_writer`` — the :class:`~py123d.api.scene.arrow.arrow_log_writer.ArrowLogWriter`,
  including a ``sync_config`` (the ``reference_column`` and ``direction`` used to
  snap modalities onto a common timeline).
* ``map_writer`` — the :class:`~py123d.api.map.arrow.arrow_map_writer.ArrowMapWriter`.

Provide a **local** config and, if you wrote a downloader, a separate **stream**
config. Keep them as distinct files — they differ in data roots and storage:

.. tab-set::

  .. tab-item:: Local (``mydataset.yaml``)

    Data is pre-downloaded; roots point at it and sensors are stored as paths.

    .. code-block:: yaml

      parser:
        _target_: py123d.parser.mydataset.mydataset_parser.MyDatasetParser
        _convert_: 'all'
        splits: ["mydataset_train", "mydataset_val"]
        mydataset_data_root: ${dataset_paths.mydataset_data_root}

      log_writer_config:
        _target_: py123d.api.scene.arrow.utils.log_writer_config.LogWriterConfig
        _convert_: 'all'
        camera_store_option: "path"   # "path", "jpeg_binary", "png_binary"
        lidar_store_option: "path"    # "path", "binary"
        lidar_codec: null
        async_conversion: ${async_conversion}
        # ...

  .. tab-item:: Stream (``mydataset-stream.yaml``)

    Roots are ``null`` (auto-detected in a temp dir); a ``downloader`` block is
    added and sensors are stored as **binary** (temp files vanish after
    conversion).

    .. code-block:: yaml

      parser:
        _target_: py123d.parser.mydataset.mydataset_parser.MyDatasetParser
        _convert_: 'all'
        splits: ["mydataset_train", "mydataset_val"]
        mydataset_data_root: null     # auto-detected
        downloader:
          _target_: py123d.parser.mydataset.mydataset_download.MyDatasetDownloader
          _convert_: 'all'
          output_dir: null            # temp dir assigned at runtime
          splits: ${dataset.parser.splits}
          dry_run: false

      log_writer_config:
        # ...
        camera_store_option: "jpeg_binary"
        lidar_store_option: "binary"
        lidar_codec: "laz"

.. note::
  When you override ``iter_modalities_async``, enable it per-config by setting the
  top-level ``async_conversion: True`` (consumed via
  ``log_writer_config.async_conversion``). The ``sync_config`` in ``log_writer``
  still controls how synchronized frames are referenced when writing.


6. Documentation
----------------

Finally, document the dataset so users know how to obtain, install, and convert
it:

#. Add a page ``docs/datasets/<name>.rst`` — mirror an existing one such as
   ``docs/datasets/nuscenes.rst``. The established structure is: an *Overview*
   dropdown (paper, download, license, splits), a *Sensor Overview* figure,
   an *Available Modalities* table, *Download*, *Installation* (the pip extra),
   *Conversion* (local + streaming), any dataset-specific sections, *Dataset
   Issues*, and *Citation*.
#. Register the page in the ``docs/datasets/index.rst`` toctree.
#. (Optional) Add a sensor-overview timeplot SVG under
   ``docs/_static/timeplots/`` and reference it with a ``.. figure::`` directive,
   as the other dataset pages do.
#. Build the docs locally and confirm the page renders without warnings before
   opening your PR.


Checklist
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Step
     - Location
   * - ``BaseDatasetParser`` subclass
     - ``src/py123d/parser/<dataset>/<dataset>_parser.py``
   * - ``BaseLogParser`` subclass
     - same package
   * - ``BaseMapParser`` subclass (if maps)
     - same package
   * - Sensor I/O loader (if new format)
     - ``src/py123d/common/io/{camera,lidar,radar}/``
   * - Optional dependency extra + gating
     - ``pyproject.toml`` + ``check_dependencies(...)``
   * - Downloader (optional)
     - ``src/py123d/parser/<dataset>/<dataset>_download.py``
   * - Hydra config pair
     - ``src/py123d/script/config/conversion/dataset/<name>{,-stream}.yaml``
   * - Dataset doc page + index entry
     - ``docs/datasets/<name>.rst`` + ``docs/datasets/index.rst``

Thank you for contributing a dataset to 123D! If anything in this guide is unclear
or you hit a case it doesn't cover, please open an issue or PR.
