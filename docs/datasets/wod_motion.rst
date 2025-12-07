Waymo Open Dataset - Motion
---------------------------

The Waymo Open Dataset (WOD) is a collective term for publicly available datasets from Waymo.
The *Motion Dataset*, abbreviated as WOD-Motion,

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `Scalability in Perception for Autonomous Driving: Waymo Open Dataset <https://arxiv.org/abs/1912.04838>`_
    * - :octicon:`download` Download
      - `waymo.com/open <https://waymo.com/open/>`_
    * - :octicon:`mark-github` Code
      - `waymo-open-dataset <https://github.com/waymo-research/waymo-open-dataset>`_
    * - :octicon:`law` License
      -
        `Waymo Dataset License Agreement for Non-Commercial Use <https://waymo.com/open/terms/>`_

        Apache License 2.0 + `Code Specific Licenses <https://github.com/waymo-research/waymo-open-dataset/blob/master/LICENSE>`_

    * - :octicon:`database` Available splits
      - ``wodp_train``, ``wodp_val``, ``wodp_test``


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 5 75

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓
     - State of the ego vehicle, including poses, and vehicle parameters, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - (✓)
     - The HD-Maps are in 3D, but may have artifacts due to polyline to polygon conversion (see below). For more information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓
     - The bounding boxes are available with the :class:`~py123d.conversion.registry.WODBoxDetectionLabel`. For more information, :class:`~py123d.datatypes.detections.BoxDetectionWrapper`.
   * - Traffic Lights
     - X
     - n/a
   * - Pinhole Cameras
     - ✓
     -
      Includes 5 cameras, see :class:`~py123d.datatypes.sensors.PinholeCamera`:

      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_F0` (front_camera)
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_L0` (front_left_camera)
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_R0` (front_right_camera)
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_L1` (left_camera)
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_R1` (right_camera)

   * - Fisheye Cameras
     - X
     - n/a
   * - LiDARs
     - ✓
     -
      Includes 5 LiDARs, see :class:`~py123d.datatypes.sensors.LiDAR`:

      - :class:`~py123d.datatypes.sensors.LiDARType.LIDAR_TOP` (top)
      - :class:`~py123d.datatypes.sensors.LiDARType.LIDAR_FRONT` (front)
      - :class:`~py123d.datatypes.sensors.LiDARType.LIDAR_SIDE_LEFT` (side_left)
      - :class:`~py123d.datatypes.sensors.LiDARType.LIDAR_SIDE_RIGHT` (side_right)
      - :class:`~py123d.datatypes.sensors.LiDARType.LIDAR_BACK` (rear)

.. dropdown:: Dataset Specific


  .. autoclass:: py123d.conversion.registry.WODBoxDetectionLabel
    :members:
    :no-inherited-members:

  .. autoclass:: py123d.conversion.registry.WODPLiDARIndex
    :members:
    :no-inherited-members:



Download
~~~~~~~~

To download the Waymo Open Dataset for Perception, please visit the `official website <https://waymo.com/open/>`_ and follow the instructions provided there.
You will need to register and download the Perception Dataset ``V1.4.3``.
(We currently do not support ``V2.0.1`` due to the missing maps.)
The expected directory structure after downloading and extracting the dataset is as follows:

.. code-block:: text

  $WODP_DATA_ROOT
    ├── testing/
    |   ├── segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord
    |   ├── ...
    |   └── segment-9806821842001738961_4460_000_4480_000_with_camera_labels.tfrecord
    ├── training/
    |   ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
    |   ├── ...
    |   └── segment-9985243312780923024_3049_720_3069_720_with_camera_labels.tfrecord
    └── validation/
        ├── segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
        ├── ...
        └── segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord

You can add the dataset root directory to the environment variable ``WODP_DATA_ROOT`` for easier access.

.. code-block:: bash

   export WODP_DATA_ROOT=/path/to/wodp_dataset_root

Optionally, you can adjust the ``py123d/script/config/common/default_dataset_paths.yaml`` accordingly.

Installation
~~~~~~~~~~~~

The Waymo Open Dataset requires additional dependencies that are included as optional dependencies in ``py123d``. You can install them via:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[waymo]

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[waymo]

These dependencies are notoriously difficult to install due to compatibility issues.
We recommend using a dedicated conda environment for this purpose. Using `uv <https://docs.astral.sh/uv/>`_ can significantly speed up the installation.
Here is an example of how to set it up:

.. code-block:: bash

  conda create -n py123d_waymo python=3.10
  conda activate py123d_waymo
  uv pip install -e .[waymo]
  # If something goes wrong: conda deactivate; conda remove -n py123d_waymo --all

You only need the Waymo Open Dataset specific dependencies if you convert the dataset or read from the raw TFRecord files.
After conversion, you may use any other ``py123d`` installation.


Dataset Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~


* **Map:** The HD-Map in Waymo has bugs ...

Citation
~~~~~~~~

If you use this dataset in your research, please cite:

.. code-block:: bibtex

  @inproceedings{Sun2020CVPR,
    title={Scalability in perception for autonomous driving: Waymo open dataset},
    author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    pages={2446--2454},
    year={2020}
  }
