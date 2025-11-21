KITTI-360
---------

The KITTI-360 dataset is an extension of the popular KITTI dataset, designed for various perception tasks in autonomous driving.
The dataset includes 9 logs (called "sequences") of varying length with stereo cameras, fisheye cameras, LiDAR data, 3D primitives, and semantic annotations.

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
     - ✓
     - State of the ego vehicle, including poses, dynamic state, and vehicle parameters, see :class:`~py123d.datatypes.vehicle_state.EgoStateSE3`.
   * - Map
     - ✓
     - The maps are in 3D vector format and defined per log, see :class:`~py123d.api.MapAPI`. The map does not include lane-level information.
   * - Bounding Boxes
     - ✓
     - The bounding boxes are available and labeled with :class:`~py123d.conversion.registry.KITTI360BoxDetectionLabel`. For further information, see :class:`~py123d.datatypes.detections.BoxDetectionWrapper`.
   * - Traffic Lights
     - X
     - n/a
   * - Pinhole Cameras
     - ✓
     - The dataset has two :class:`~py123d.datatypes.sensors.PinholeCamera` in a stereo setup:

       - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_STEREO_L` (image_00)
       - :class:`~py123d.datatypes.sensors.PinholeCameraType.PCAM_STEREO_R` (image_01)

   * - Fisheye Cameras
     - ✓
     - The dataset has two :class:`~py123d.datatypes.sensors.FisheyeMEICamera`:

       - :class:`~py123d.datatypes.sensors.FisheyeMEICameraType.FCAM_L` (image_02)
       - :class:`~py123d.datatypes.sensors.FisheyeMEICameraType.FCAM_R` (image_03)
   * - LiDARs
     - ✓
     - The dataset has :class:`~py123d.datatypes.sensors.LiDAR` mounted on the roof:

       - :class:`~py123d.datatypes.sensors.LiDARType.LIDAR_TOP` (velodyne_points)

.. dropdown:: Dataset Specific

  .. autoclass:: py123d.conversion.registry.KITTI360BoxDetectionLabel
    :members:
    :no-index:
    :no-inherited-members:

  .. autoclass:: py123d.conversion.registry.KITTI360LiDARIndex
    :members:
    :no-index:
    :no-inherited-members:


Download
~~~~~~~~

You can download the KITTI-360 dataset from the `official website <https://www.cvlibs.net/datasets/kitti-360/>`_. Please follow the instructions provided there to obtain the data.
The 123D library supports expect the dataset in the following directory structure:

.. code-block:: text

  $KITTI360_DATA_ROOT/
  ├── calibration/
  │   ├── calib_cam_to_pose.txt
  │   ├── calib_cam_to_velo.txt
  │   ├── calib_sick_to_velo.txt
  │   ├── image_02.yaml
  │   ├── image_03.yaml
  │   └── perspective.txt
  ├── data_2d_raw/
  │   ├── 2013_05_28_drive_0000_sync/
  │   │   ├── image_00/
  │   │   │   ├── data_rect
  │   │   │   │   ├── 0000000000.png
  │   │   │   │   ├── ...
  │   │   │   │   └── 0000011517.png
  │   │   │   └── timestamps.txt
  │   │   ├── image_01/
  │   │   │   └── ...
  │   │   ├── image_02/
  │   │   │   ├── data_rgb
  │   │   │   │   ├── 0000000000.png
  │   │   │   │   ├── ...
  │   │   │   │   └── 0000011517.png
  │   │   │   └── timestamps.txt
  │   │   └── image_03/
  │   │       └── ...
  │   ├── ...
  │   └── 2013_05_28_drive_0018_sync/
  │       └── ...
  ├── data_2d_semantics/ (not yet supported)
  │   └── ...
  ├── data_3d_bboxes/
  │   ├── train
  │   │   ├── 2013_05_28_drive_0000_sync.xml
  │   │   ├── ...
  │   │   └── 2013_05_28_drive_0010_sync.xml
  │   └── train_full
  │       ├── 2013_05_28_drive_0000_sync.xml
  │       ├── ...
  │       └── 2013_05_28_drive_0010_sync.xml
  ├── data_3d_raw/
  │   ├── 2013_05_28_drive_0000_sync/
  │   │   └── velodyne_points/
  │   │       ├── data
  │   │       │   ├── 0000000000.bin
  │   │       │   ├── ...
  │   │       │   └── 0000011517.bin
  │   │       └── timestamps.txt
  │   ├── ...
  │   └── 2013_05_28_drive_0018_sync/
  │       └── ...
  ├── data_3d_semantics/ (not yet supported)
  │   └── ...
  └── data_poses/
      ├── 2013_05_28_drive_0000_sync/
      │   ├── cam0_to_world.txt
      │   ├── oxts/
      │   │   └── ...
      │   └── poses.txt
      ├── ...
      └── 2013_05_28_drive_0018_sync/
          └── ...

Note that not all data modalities are currently supported in 123D. For example, semantic 2D and 3D data are not yet integrated.


Installation
~~~~~~~~~~~~

No additional installation steps are required beyond the standard `py123d`` installation.


Conversion
~~~~~~~~~~

You can convert the KITTI-360 dataset by running:

.. code-block:: bash

  py123d-conversion datasets=["kitti360_dataset"]


Note, that you can assign the logs of KITTI-360 to different splits (e.g., "train", "val", "test") in the ``kitti360_dataset.yaml`` config.


Dataset Issues
~~~~~~~~~~~~~~

* **Ego Vehicle:** The vehicle parameters from the VW station wagon are partially estimated and may be subject to inaccuracies.
* **Map:** The ground primitives in KITTI-360 only cover surfaces, e.g. of the road, but not lane-level information. Drivable areas, road edges, walkways, driveways are included.
* **Bounding Boxes:** Bounding boxes in KITTI-360 annotated globally. We therefore determine which boxes are visible in each frame on the number of LiDAR points contained in the box.


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
