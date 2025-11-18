nuScenes
--------

The nuScenes dataset is multi-modal autonomous driving dataset that includes data from cameras, LiDARs, and radars, along with detailed annotations from Boston and Singapore.
In total, the dataset contains 1000 driving logs, each of 20 second duration, resulting in 5.5 hours of data.
All logs include ego-vehicle data, camera images, LiDAR point clouds, bounding boxes, and map data.


.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Papers
      -
        `nuscenes: A multimodal dataset for autonomous driving <https://arxiv.org/abs/1903.11027>`_
    * - :octicon:`download` Download
      - `nuscenes.org <https://www.nuscenes.org/>`_
    * - :octicon:`mark-github` Code
      - `nuscenes-devkit <https://github.com/nutonomy/nuscenes-devkit>`_
    * - :octicon:`law` License
      -
        `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_

        `nuScenes Terms of Use <https://www.nuscenes.org/terms-of-use>`_

        Apache License 2.0
    * - :octicon:`database` Available splits
      - ``nuscenes_train``, ``nuscenes_val``, ``nuscenes_test``, ``nuscenes-mini_train``, ``nuscenes-mini_val``, ``nuscenes-mini_test``


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
     - State of the ego vehicle, including poses, dynamic state, and vehicle parameters, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - (✓)
     - The HD-Maps are in 2D vector format and defined per-location. For more information, see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓
     - The bounding boxes are available with the :class:`~py123d.conversion.registry.NuScenesBoxDetectionLabel`. For more information, see :class:`~py123d.datatypes.detections.BoxDetectionWrapper`.
   * - Traffic Lights
     - X
     -
   * - Pinhole Cameras
     - ✓
     -
      nuScenes includes 6x :class:`~py123d.datatypes.sensors.PinholeCamera`:

      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_F0`: CAM_FRONT
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_R0`: CAM_FRONT_RIGHT
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_R1`: CAM_BACK_RIGHT
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_L0`: CAM_FRONT_LEFT
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_L1`: CAM_BACK_LEFT
      - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_B0`: CAM_BACK
   * - Fisheye Cameras
     - X
     -
   * - LiDARs
     - ✓
     - nuScenes has one :class:`~py123d.datatypes.sensors.LiDAR` of type :class:`~py123d.datatypes.sensors.LiDARType.LIDAR_TOP`.
.. dropdown:: Dataset Specific

  .. autoclass:: py123d.conversion.registry.NuScenesBoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:

  .. autoclass:: py123d.conversion.registry.NuScenesLiDARIndex
    :members:
    :no-index:
    :no-inherited-members:


Download
~~~~~~~~

You need to install the nuScenes dataset from the `official website <https://www.nuscenes.org/download>`_.
The 123D conversion expects the following directory structure:

.. code-block:: none

  $NUSCENES_DATA_ROOT
    ├── can_bus/
    │   ├── scene-0001_meta.json
    │   ├── ...
    │   └── scene-1110_zoe_veh_info.json
    ├── maps/
    │   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    │   ├── ...
    │   ├── 93406b464a165eaba6d9de76ca09f5da.png
    │   ├── basemap/
    │   │   └── ...
    │   ├── expansion/
    │   │   └── ...
    │   └── prediction/
    │       └── ...
    ├── samples/
    │   ├── CAM_BACK/
    │   │   └── ...
    │   ├── ...
    │   └── RADAR_FRONT_RIGHT/
    │       └── ...
    ├── sweeps/
    │   └── ...
    ├── v1.0-mini/
    │   ├── attribute.json
    │   ├── ...
    │   └── visibility.json
    ├── v1.0-test/
    │   ├── attribute.json
    │   ├── ...
    │   └── visibility.json
    └── v1.0-trainval/
        ├── attribute.json
        ├── ...
        └── visibility.json

Lastly, you need to add the following environment variables to your ``~/.bashrc`` according to your installation paths:

.. code-block:: bash

  export NUSCENES_DATA_ROOT=/path/to/nuplan/data/root

Or configure the config ``py123d/script/config/common/default_dataset_paths.yaml`` accordingly.

Installation
~~~~~~~~~~~~

For nuScenes, additional installation that are included as optional dependencies in ``py123d`` are required. You can install them via:

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: bash

      pip install py123d[nuscenes]

  .. tab-item:: Source

    .. code-block:: bash

      pip install -e .[nuscenes]

Conversion
~~~~~~~~~~~~

You can convert the nuScenes dataset (or mini dataset) by running:

.. code-block:: bash

  py123d-conversion datasets=["nuscenes_dataset"]
  # or
  py123d-conversion datasets=["nuscenes_mini_dataset"]



Dataset Issues
~~~~~~~~~~~~~~

* **Map:** The HD-Maps are only available in 2D.
* ...


Citation
~~~~~~~~

If you use nuPlan in your research, please cite:

.. code-block:: bibtex

  @article{Caesar2020CVPR,
    title={nuscenes: A multimodal dataset for autonomous driving},
    author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    year={2020}
  }
