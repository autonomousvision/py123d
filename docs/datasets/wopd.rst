Waymo Open Perception Dataset (WOPD)
------------------------------------

.. sidebar:: WOPD

  .. image:: https://images.ctfassets.net/e6t5diu0txbw/4LpraC18sHNvS87OFnEGKB/63de105d4ce623d91cfdbc23f77d6a37/Open_Dataset_Download_Hero.jpg?fm=webp&q=90
    :alt: Dataset sample image
    :width: 290px

  | **Paper:** `Name of Paper <https://example.com/paper>`_
  | **Download:** `Documentation <https://example.com/paper>`_
  | **Code:** [Code]
  | **Documentation:** [License type]
  | **License:** [License type]
  | **Duration:** [Duration here]
  | **Supported Versions:** [Yes/No/Conditions]
  | **Redistribution:** [Yes/No/Conditions]

Description
~~~~~~~~~~~

[Provide a detailed description of the dataset here, including its purpose, collection methodology, and key characteristics.]

Installation
~~~~~~~~~~~~

[Instructions for installing or accessing the dataset]

.. code-block:: bash

   # Example installation commands
   pip install py123d[dataset_name]
   # or
   wget https://example.com/dataset.zip


.. code-block:: bash

  conda create -n py123d_waymo python=3.10
  conda activate py123d_waymo
  pip install -e .[waymo]

  # pip install protobuf==6.30.2
  # pip install tensorflow==2.13.0
  # pip install waymo-open-dataset-tf-2-12-0==1.6.6

Available Data
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 5 70


   * - **Name**
     - **Available**
     - **Description**
   * - Ego Vehicle
     - X
     - [Description of ego vehicle data]
   * - Map
     - X
     - [Description of ego vehicle data]
   * - Bounding Boxes
     - X
     - [Description of ego vehicle data]
   * - Traffic Lights
     - X
     - [Description of ego vehicle data]
   * - Cameras
     - X
     - [Description of ego vehicle data]
   * - LiDARs
     - X
     - [Description of ego vehicle data]

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

  @inproceedings{Sun2020CVPR,
    title={Scalability in perception for autonomous driving: Waymo open dataset},
    author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    pages={2446--2454},
    year={2020}
  }
