nuPlan
-----

.. sidebar:: nuPlan

  .. image:: https://www.nuplan.org/static/media/nuPlan_final.3fde7586.png
    :alt: Dataset sample image
    :width: 290px

  | **Paper:** `Towards learning-based planning:The nuPlan benchmark for real-world autonomous driving <https://arxiv.org/abs/2403.04133>`_
  | **Download:** `www.nuscenes.org/nuplan <https://www.nuscenes.org/nuplan>`_
  | **Code:** `www.github.com/motional/nuplan-devkit <https://github.com/motional/nuplan-devkit>`_
  | **Documentation:** `nuPlan Documentation <https://nuplan-devkit.readthedocs.io/>`_
  | **License:** `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>`_, `nuPlan Dataset License <https://www.nuscenes.org/terms-of-use>`_
  | **Duration:** 1282 hours (120 hours of sensor data)
  | **Supported Versions:** [TODO]
  | **Redistribution:** [TODO]

Description
~~~~~~~~~~~

[Provide a detailed description of the dataset here, including its purpose, collection methodology, and key characteristics.]

Installation
~~~~~~~~~~~~

[Instructions for installing or accessing the dataset]

.. code-block:: bash

   # Example installation commands
   pip install d123[dataset_name]
   # or
   wget https://example.com/dataset.zip

Available Data
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 70

   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - ✓
     - [Description of ego vehicle data]
   * - Map
     - ✓
     - [Description of map data]
   * - Bounding Boxes
     - X
     - [Description of bounding boxes data]
   * - Traffic Lights
     - X
     - [Description of traffic lights data]
   * - Cameras
     - X
     - [Description of cameras data]
   * - LiDARs
     - X
     - [Description of LiDARs data]

Dataset Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~

[Document any known issues, limitations, or considerations when using this dataset]

* Issue 1: Description
* Issue 2: Description
* Issue 3: Description

Citation
~~~~~~~~


If you use this dataset in your research, please cite:

.. code-block:: bibtex

  @article{Karnchanachari2024ICRA,
    title={Towards learning-based planning: The nuplan benchmark for real-world autonomous driving},
    author={Karnchanachari, Napat and Geromichalos, Dimitris and Tan, Kok Seang and Li, Nanxiang and Eriksen, Christopher and Yaghoubi, Shakiba and Mehdipour, Noushin and Bernasconi, Gianmarco and Fong, Whye Kit and Guo, Yiluan and others},
    booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
    year={2024},
  }
  @article{Caesar2021CVPRW,
    title={nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles},
    author={Caesar, Holger and Kabzan, Juraj and Tan, Kok Seang and Fong, Whye Kit and Wolff, Eric and Lang, Alex and Fletcher, Luke and Beijbom, Oscar and Omari, Sammy},
    booktitle={Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year={2021}
  }
