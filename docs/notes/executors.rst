Executors
=========

Conversion in 123D can be slow and embarrassingly parallel: each log and each map
is processed independently. An **executor** is the abstraction that distributes
this per-log / per-map work across threads, processes, or a Ray cluster. The same
conversion code runs unchanged on a laptop core, on all cores of a single machine,
or on a multi-node SLURM cluster — only the executor configuration changes.

This page explains the available backends, how to select and tune them from a
typical Hydra script, and how to troubleshoot the common operational issues (Ray
RAM usage, subprocess safety, GPU allocation, distributed mode).


1. General
----------

Concepts
~~~~~~~~

All executors implement the abstract base class
:class:`~py123d.common.execution.executor.Executor`. Work is submitted as a
:class:`~py123d.common.execution.executor.Task` (a callable plus optional
``num_cpus`` / ``num_gpus`` resource hints) and mapped over a list of arguments
via :meth:`~py123d.common.execution.executor.Executor.map`. Each backend declares
the resources it has available through
:class:`~py123d.common.execution.executor.ExecutorResources` (number of nodes,
CPUs per node, GPUs per node).

The framework constructs lightweight, **picklable** per-log and per-map handles on
the main process and then hands them to the executor's workers, which perform the
heavy I/O lazily. See :doc:`/notes/adding_datasets` for how parsers fit into this
pipeline.

Choosing a backend
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 18 12 22 10 20

   * - Executor
     - Backend
     - Nodes
     - Threading model
     - GPU
     - Typical use case
   * - ``sequential_executor``
     - none
     - 1
     - synchronous / blocking
     - no
     - debugging, profiling, reproducing errors
   * - ``thread_pool_executor``
     - ``concurrent.futures``
     - 1
     - multi-threaded (shared GIL)
     - no
     - I/O-bound work, moderate parallelism
   * - ``process_pool_executor``
     - ``concurrent.futures``
     - 1
     - multi-process (``forkserver``)
     - no
     - CPU-bound work on a single machine
   * - ``ray_executor``
     - `Ray <https://www.ray.io/>`_
     - 1..N
     - distributed tasks / actors
     - yes (fractional)
     - large-scale conversion, clusters, GPU jobs

``ray_executor`` is the **default**. Use ``sequential_executor`` whenever you need
deterministic, debuggable behaviour or a readable traceback.

Selecting an executor with Hydra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The active executor is chosen by the ``execution`` config group. The default is
set in ``src/py123d/script/config/common/default_common.yaml``:

.. code-block:: yaml

   defaults:
     - execution: ray_executor

Override it (and any of its fields) directly on the command line:

.. code-block:: bash

   # Debug single-threaded
   py123d-convert dataset=nuscenes execution=sequential_executor

   # Single machine, 4 worker processes
   py123d-convert dataset=nuscenes execution=process_pool_executor execution.max_workers=4

   # Ray, capped at 16 CPU threads per node
   py123d-convert dataset=nuscenes execution=ray_executor execution.threads_per_node=16

How a script uses it
~~~~~~~~~~~~~~~~~~~~~

A script instantiates the configured backend with ``build_executor(cfg)`` and then
distributes work with the helpers in :mod:`py123d.common.execution.utils`. For
example, :mod:`py123d.script.run_conversion` does:

.. code-block:: python

   executor = build_executor(cfg)
   executor_map_chunked_list(
       executor,
       partial(convert_fn, cfg=cfg),
       dataset_parser.get_log_parsers(),
       name=f"Logs {parser_class_name}",
   )

Troubleshooting
~~~~~~~~~~~~~~~

.. warning::
   **Ray uses a lot of RAM.** Ray serializes task arguments and results through its
   object store, and every worker holds its own copy of the per-log handles it is
   processing. With many threads this multiplies memory use, and large results can
   spill to disk. If you hit out-of-memory:

   * Lower ``execution.threads_per_node`` to run fewer concurrent workers.
   * Switch to ``execution=sequential_executor`` to confirm the problem is
     concurrency and to get a clean traceback.
   * Keep per-log handles small and lazy — do the heavy reads inside the worker,
     not on the main process.

.. note::
   **"Ray is running, we will shut it down before starting again!"** Ray cannot be
   re-initialized while already running, so the executor shuts down any existing
   instance before starting. This is expected when several executors are created in
   the same process (e.g. in tests).

.. note::
   **ProcessPool uses the** ``forkserver`` **start method**, not ``fork`` (unsafe
   with threads / open handles) or ``spawn`` (slow). If a child process fails to
   start, check that everything captured by the mapped function is picklable.

.. warning::
   **Fractional GPUs are not validated.** When a :class:`Task` requests
   ``num_gpus < 1``, Ray allocates only a fraction of a GPU. It is the user's
   responsibility to ensure the model actually fits into that fraction of GPU
   memory.

.. note::
   **Distributed / SLURM mode** (``execution.use_distributed=true``) expects the
   cluster environment variables ``ip_head``, ``redis_password`` and ``num_nodes``
   to be set, or a ``master_node_ip`` to connect to a running cluster remotely.
   Local single-machine mode is the default and needs none of these.

.. tip::
   Ray suppresses noisy ``botocore`` credential logs that would otherwise be
   printed repeatedly during serialization. This is intentional and does not
   indicate a problem.


2. Sequential executor
----------------------

.. code-block:: yaml

   _target_: py123d.common.execution.sequential_executor.SequentialExecutor
   _convert_: 'all'

.. autoclass:: py123d.common.execution.sequential_executor.SequentialExecutor
   :members:
   :show-inheritance:


3. Thread pool executor
-----------------------

.. code-block:: yaml

   _target_: py123d.common.execution.thread_pool_executor.ThreadPoolExecutor
   _convert_: 'all'
   max_workers: null  # Number of threads to use, "null" means all available CPUs

.. autoclass:: py123d.common.execution.thread_pool_executor.ThreadPoolExecutor
   :members:
   :show-inheritance:


4. Process pool executor
------------------------

.. code-block:: yaml

   _target_: py123d.common.execution.process_pool_executor.ProcessPoolExecutor
   _convert_: 'all'
   max_workers: null  # Number of processes to use, "null" means all available CPUs

.. autoclass:: py123d.common.execution.process_pool_executor.ProcessPoolExecutor
   :members:
   :show-inheritance:


5. Ray executor
---------------

.. code-block:: yaml

   _target_: py123d.common.execution.ray_executor.RayExecutor
   _convert_: 'all'
   master_node_ip: null    # Set to a master node IP to connect to a cluster remotely
   threads_per_node: null  # CPU threads to use per node, "null" means all available
   log_to_driver: true     # If true, printouts from Ray workers are shown in the driver
   logs_subdir: 'logs'     # Subdirectory for logs inside the experiment directory
   use_distributed: false  # Whether to use Ray's built-in distributed mode

Field reference:

* ``master_node_ip`` — connect to an already-running Ray cluster at this IP instead
  of starting a local instance.
* ``threads_per_node`` — cap the number of CPU threads used per node; ``null`` uses
  all available. The main lever for trading throughput against RAM.
* ``log_to_driver`` — mirror worker stdout/stderr to the driver process.
* ``logs_subdir`` — where per-worker log files are written under the experiment
  directory.
* ``use_distributed`` — enable multi-node distributed mode (see the troubleshooting
  note above).

.. autoclass:: py123d.common.execution.ray_executor.RayExecutor
   :members:
   :show-inheritance:

.. automodule:: py123d.common.execution.ray_utils
   :members:
   :no-imported-members:


6. Utilities
------------

Two mapping strategies are provided in
:mod:`py123d.common.execution.utils`. Pick based on how uniform your task
runtimes are:

* :func:`~py123d.common.execution.utils.executor_map_chunked_list` —
  **pre-chunks** the input into one chunk per worker. Low scheduling overhead, but
  uses **static** load distribution. Best when tasks take a similar amount of time.
* :func:`~py123d.common.execution.utils.executor_map_queued` — submits each item as
  an individual task so idle workers dynamically pick up the next one. Better
  **load balancing** for uneven task runtimes, at the cost of higher scheduling
  overhead.

.. automodule:: py123d.common.execution.utils
   :members:
   :no-imported-members:
