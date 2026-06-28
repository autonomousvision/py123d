.. _nureasoning:

nuReasoning
-----------

nuReasoning is a reasoning-centric autonomous driving dataset and benchmark focused on
long-tail scenarios.
It contains 20,000 clips of 20 seconds each (~105 hours), collected from multiple cities,
combining synchronized multi-view camera images, LiDAR point clouds, ego state, HD maps,
and traffic-signal context with human-verified reasoning annotations.
The reasoning annotations span three categories — spatial reasoning, driving decisions,
and counterfactual reasoning — and support both a Reasoning VQA benchmark and a planning
benchmark.

.. note::
  py123d currently exposes only the **mini** subset of nuReasoning
  (parts 1–3 of the HuggingFace ``train`` split).


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Papers
      -
        `nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving <https://arxiv.org/abs/2605.31572>`_

        `Project page <https://nureasoning.github.io/>`_
    * - :octicon:`download` Download
      - `qixuewei/nuReasoning on HuggingFace <https://huggingface.co/datasets/qixuewei/nuReasoning>`_
    * - :octicon:`mark-github` Code
      - A public devkit is announced as *Coming Soon* on the `project page <https://nureasoning.github.io/>`_.
    * - :octicon:`law` License
      - Apache License 2.0
    * - :octicon:`database` Available splits
      - ``nureasoning-mini_train``


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 70

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓
     - State of the ego vehicle, including poses, dynamic state, and vehicle parameters, see :class:`~py123d.datatypes.EgoStateSE3`.
   * - Map
     - (✓)
     - The HD-Maps are in 2D vector format and stored per-log (one map per clip). For more information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓
     - The bounding boxes are available with the :class:`~py123d.parser.registry.NureasoningBoxDetectionLabel`. For more information, see :class:`~py123d.datatypes.BoxDetectionsSE3`.
   * - Traffic Lights
     - ✓
     - Traffic-signal states are provided per frame.
   * - Cameras
     - ✓
     -
      nuReasoning includes 8x :class:`~py123d.datatypes.Camera`:

      - :class:`~py123d.datatypes.CameraID.PCAM_F0`: front
      - :class:`~py123d.datatypes.CameraID.PCAM_B0`: back
      - :class:`~py123d.datatypes.CameraID.PCAM_L0`: front_left
      - :class:`~py123d.datatypes.CameraID.PCAM_L1`: left
      - :class:`~py123d.datatypes.CameraID.PCAM_L2`: back_left
      - :class:`~py123d.datatypes.CameraID.PCAM_R0`: front_right
      - :class:`~py123d.datatypes.CameraID.PCAM_R1`: right
      - :class:`~py123d.datatypes.CameraID.PCAM_R2`: back_right

   * - Lidars
     - ✓
     -
      A single merged :class:`~py123d.datatypes.Lidar` point cloud fusing five sensors
      (top, front, side-left, side-right, back).
      The source sensor of each point is encoded in a ``lidar_info`` channel, and the
      cloud additionally carries ``ring``, ``intensity``, ``azimuth``, ``range`` and
      return information. Point clouds are stored as LZF-compressed PCD files.
   * - Reasoning
     - ✓
     - Human-verified spatial, decision, and counterfactual reasoning annotations are
       available per frame. They are currently passed through as the raw nuReasoning
       reasoning JSON (no dedicated py123d datatype yet).

.. dropdown:: Dataset Specific

  .. autoclass:: py123d.parser.registry.NureasoningBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:

  .. note::
    nuReasoning does not have a published object taxonomy yet — the label set above is
    provisional and may be incomplete or change in a future release.

Download
~~~~~~~~

nuReasoning is hosted as a public, Apache-2.0 licensed dataset on HuggingFace at
`qixuewei/nuReasoning <https://huggingface.co/datasets/qixuewei/nuReasoning>`_.
py123d ships an automated downloader that fetches and extracts the per-clip archives for
you.

.. code-block:: bash

  # Download the configured selection into $NUREASONING_DATA_ROOT
  py123d-download dataset=nureasoning

The downloader exposes several selection knobs (see
``py123d/script/config/download/dataset/nureasoning.yaml``):

* ``splits`` — e.g. ``[train]``, ``[train, validation]`` (``null`` discovers all live from the repo)
* ``parts`` — e.g. ``[part_1, part_2]``
* ``log_names`` — explicit clip names ``<log>_<token>`` (mutually exclusive with ``num_logs``)
* ``num_logs`` — the first N clips of the selection (or N random with ``sample_random=true`` and ``seed``)
* ``max_workers`` — parallel clip download/extract workers (default ``8``)
* ``keep_zip`` — keep each ``.zip`` next to its extracted directory (default: extract then discard)

Each selected clip is downloaded as a single ``.zip`` and extracted to
``<output_dir>/<split>/<part>/<clip>/``. The 123D conversion expects the following
directory structure:

.. code-block:: none

  $NUREASONING_DATA_ROOT
    └── train/
        ├── part_1/
        │   ├── <log_name>_<keyframe_token>/
        │   │   ├── metadata.json
        │   │   ├── map.pkl
        │   │   ├── ego_state/
        │   │   │   └── <timestamp_us>.pkl
        │   │   ├── annotations/
        │   │   │   └── <timestamp_us>.pkl
        │   │   ├── reasoning/
        │   │   │   └── <timestamp_us>.json
        │   │   ├── cameras/
        │   │   │   ├── front.jpg
        │   │   │   ├── back.jpg
        │   │   │   └── ...
        │   │   └── lidar/
        │   │       └── <timestamp_us>.pcd
        │   └── ...
        ├── part_2/
        └── part_3/

Lastly, you need to add the following environment variable to your ``~/.bashrc`` according
to your installation path:

.. code-block:: bash

  export NUREASONING_DATA_ROOT=/path/to/nureasoning/data/root

Or configure the config ``py123d/script/config/common/default_dataset_paths.yaml`` accordingly.

Installation
~~~~~~~~~~~~

Downloading and streaming nuReasoning requires the HuggingFace Hub client, included as an
optional dependency in ``py123d``. You can install it via:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[hf]

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[hf]

Conversion
~~~~~~~~~~~~

**Local mode** — data already extracted to ``$NUREASONING_DATA_ROOT`` (see the `Download`_
section above):

.. code-block:: bash

  py123d-conversion dataset=nureasoning-mini

.. note::
  The local conversion of nuReasoning by default does not store sensor data in the logs,
  but only relative file paths (``camera_store_option: "path"`` and
  ``lidar_store_option: "path"``), which are resolved against the nuReasoning sensor root
  at read time. To change this behavior, adapt the ``nureasoning-mini.yaml`` converter
  configuration.

**Streaming mode** — materialize the selected clips from the HuggingFace repo into a
session-scoped temp directory at parser construction time, convert from it, and delete the
temp directory afterwards. The mini subset corresponds to parts 1–3 of the HuggingFace
``train`` split.

.. code-block:: bash

  py123d-conversion dataset=nureasoning-mini-stream

The repo is public, so no token is required; if needed, ``hf_token`` falls back to
``$HF_TOKEN`` / ``$HUGGINGFACE_HUB_TOKEN``.

.. note::
  Streaming mode forces ``camera_store_option: "jpeg_binary"`` and
  ``lidar_store_option: "binary"`` (with the ``laz`` codec) — the temp directory is
  deleted immediately after conversion, so any ``"path"`` references would point at
  vanished sources.


Citation
~~~~~~~~

If you use nuReasoning in your research, please cite:

.. code-block:: bibtex

  @misc{huang2026nureasoning,
    title={nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving},
    author={Huang, Zhiyu and Liu, Johnson and Song, Rui and others},
    year={2026},
    eprint={2605.31572},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
