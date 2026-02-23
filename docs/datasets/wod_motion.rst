Waymo Open Dataset - Motion
---------------------------

The Waymo Open Dataset (WOD) is a collective term for publicly available datasets from Waymo.

.. warning::

   The WOD-Motion dataset is not fully supported yet. You might encounter issues. This documentation is incomplete.

.. dropdown:: Overview
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset <https://arxiv.org/abs/2104.10133>`_
    * - :octicon:`download` Download
      - `waymo.com/open <https://waymo.com/open/>`_
    * - :octicon:`mark-github` Code
      - `waymo-open-dataset <https://github.com/waymo-research/waymo-open-dataset>`_
    * - :octicon:`law` License
      -
        `Waymo Dataset License Agreement for Non-Commercial Use <https://waymo.com/open/terms/>`_

        Apache License 2.0 + `Code Specific Licenses <https://github.com/waymo-research/waymo-open-dataset/blob/master/LICENSE>`_

    * - :octicon:`database` Available splits
      - ``wod-motion_train``, ``wod-motion_val``, ``wod-motion_test``


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
     - The bounding boxes are available with the :class:`~py123d.conversion.registry.WODMotionBoxDetectionLabel`. For more information, :class:`~py123d.datatypes.detections.BoxDetectionWrapper`.
   * - Traffic Lights
     - ✓
     - Traffic lights include the status and the lane id they are associated with, see :class:`~py123d.datatypes.detections.TrafficLightDetectionWrapper`.
   * - Pinhole Cameras
     - X
     - n/a
   * - Fisheye Cameras
     - X
     - n/a
   * - Lidars
     - X
     - n/a

.. dropdown:: Dataset Specific


  .. autoclass:: py123d.conversion.registry.WODMotionBoxDetectionLabel
    :members:
    :no-inherited-members:


Download
~~~~~~~~

To download the Waymo Open Dataset - Motion, please visit the `official website <https://waymo.com/open/>`_.


.. code-block:: text

  $WOD_MOTION_DATA_ROOT
    ├── testing/
    |   ├── ...
    |   ├── ...
    |   └── ...
    ├── training/
    |   ├── ...
    |   ├── ...
    |   └── ...
    └── validation/
        ├── ...
        ├── ...
        └── ...

You can add the dataset root directory to the environment variable ``WOD_MOTION_DATA_ROOT`` for easier access.

.. code-block:: bash

   export WOD_MOTION_DATA_ROOT=/path/to/wod_motion_data_root

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

  @inproceedings{Ettinger2021ICCV,
    title={Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset},
    author={Ettinger, Scott and Cheng, Shuyang and Caine, Benjamin and Liu, Chenxi and Zhao, Hang and Pradhan, Sabeek and Chai, Yuning and Sapp, Ben and Qi, Charles R and Zhou, Yin and others},
    booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
    pages={9710--9719},
    year={2021}
  }
