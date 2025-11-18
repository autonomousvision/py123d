KITTI-360
---------

.. dropdown:: Quick Links
  :open:

  .. list-table::
    :header-rows: 0
    :widths: 20 60

    * -
      -
    * - :octicon:`file` Paper
      - `KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D <https://arxiv.org/abs/2109.13410>`_
    * - :octicon:`download` Download
      - `cvlibs.net/datasets/kitti-360 <https://www.cvlibs.net/datasets/kitti-360/>`_
    * - :octicon:`mark-github` Code
      - `github.com/autonomousvision/kitti360scripts <https://github.com/autonomousvision/kitti360scripts>`_
    * - :octicon:`law` License
      -
        - `CC BY-NC-SA 3.0 <https://creativecommons.org/licenses/by-nc-sa/3.0/>`_
        - MIT License
    * - :octicon:`database` Available splits
      - n/a


Available Modalities
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 70

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓ / (✓) / X
     - ..., see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - ✓ / (✓) / X
     - ..., see :class:`~py123d.api.MapAPI`.
   * - Bounding Boxes
     - ✓ / (✓) / X
     - ..., see :class:`~py123d.datatypes.detections.BoxDetectionWrapper`.
   * - Traffic Lights
     - ✓ / (✓) / X
     - ..., see :class:`~py123d.datatypes.detections.TrafficLightDetectionWrapper`.
   * - Pinhole Cameras
     - ✓ / (✓) / X
     - ..., see :class:`~py123d.datatypes.sensors.PinholeCamera`.
   * - Fisheye Cameras
     - ✓ / (✓) / X
     - ..., see :class:`~py123d.datatypes.sensors.FisheyeCamera`.
   * - LiDARs
     - ✓ / (✓) / X
     - ..., see :class:`~py123d.datatypes.sensors.LiDAR`.


Download
~~~~~~~~

...

The 123D conversion expects the following directory structure:

Installation
~~~~~~~~~~~~

For *Template*, additional installation that are included as optional dependencies in ``py123d`` are required. You can install them via:

.. code-block:: bash

  pip install py123d[template]

Or if you are installing from source:

.. code-block:: bash

  pip install -e .[template]


Dataset Specific
~~~~~~~~~~~~~~~~

.. dropdown:: Box Detection Labels

  .. autoclass:: py123d.conversion.registry.DefaultBoxDetectionLabel
    :members:
    :no-inherited-members:

.. dropdown:: LiDAR Index

  .. autoclass:: py123d.conversion.registry.DefaultLiDARIndex
    :members:
    :no-inherited-members:



Dataset Issues
~~~~~~~~~~~~~~

[Document any known issues, limitations, or considerations when using this dataset]

* Issue 1: Description
* Issue 2: Description
* Issue 3: Description


Citation
~~~~~~~~

If you use *Template* in your research, please cite:

.. code-block:: bibtex

  @article{AuthorYearConference,
    title={Template: Some Dataset for Autonomous Driving},
    author={},
    booktitle={},
    year={}
  }


Citation
~~~~~~~~

If you use KITTI-360 in your research, please cite:

.. code-block:: bibtex

  @article{Liao2022PAMI,
    title =  {{KITTI}-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D},
    author = {Yiyi Liao and Jun Xie and Andreas Geiger},
    journal = {Pattern Analysis and Machine Intelligence (PAMI)},
    year = {2022},
  }
