.. _tutorial-cli-matrix:

Tutorial: experiment matrices from the CLI
==========================================

Managing one Python script per experiment becomes impractical as the
number of experiments grows. The fl-sim command line interface runs a
whole **experiment matrix** from a single YAML file: the grid of
configurations is described once, and the CLI expands and executes every
combination. This tutorial builds a four-run matrix on synthetic data and
shows where the results are written.

The YAML syntax reference lives in :doc:`cli`; this page is the workflow
narrative. Prerequisites: :doc:`tutorial-first-experiment`.

The matrix file
---------------

Here is the complete config for this tutorial
(``example-configs/fedsynthetic-demo.yml``):

.. literalinclude:: ../../example-configs/fedsynthetic-demo.yml
   :language: yaml

Three sections matter:

- ``strategy.matrix`` — the grid. Every combination of the listed values
  is run once: 2 sampling ratios × 2 seeds = 4 experiments.
- ``${{ matrix.<key> }}`` — interpolation. Wherever this pattern appears,
  the CLI substitutes the current combination's value. ``seed`` sits at
  the top level, so it seeds the dataset *and* the server config of that
  run.
- everything else — ``algorithm`` (name plus server/client configs),
  ``dataset``, ``model`` — is exactly the config you would pass in
  Python, per :doc:`quickstart`, just spelled as YAML. Dataset-specific
  constructor arguments (``alpha``, ``beta``, ``iid``, ``num_clients``
  for :class:`~fl_sim.data_processing.fed_synthetic.FedSynthetic`) are
  passed straight through.

Running the matrix
------------------

.. code-block:: bash

    fl-sim example-configs/fedsynthetic-demo.yml

The four runs execute in sequence, each logging its combination
(``ratio=0.5 seed=0``, ...). The total time is a few seconds; nothing is
downloaded.

Results location
----------------

All runs write into ``~/.cache/fl-sim/.logs/fedsynthetic-demo/`` (the
``log_dir`` of this config). Per run you get one ``.txt`` and one
``.json`` log — the same files :doc:`quickstart` described — plus one
``config.json`` holding the exact server/client configs of the most
recent run (a previous ``config.json`` is rotated to
``config.json.bak``).

One detail matters as soon as several runs finish within the same minute:
log file names carry a timestamp **plus the run's ``tag``**. Without a
distinct tag per matrix combination, identically named logs overwrite one
another. This is the purpose of the ``tag: ratio_${{ ... }}_seed_${{ ... }}``
line in the config above: every matrix arm receives its own tag.

Beyond serial execution
-----------------------

Long experiment matrices benefit from more than one process. The
``strategy`` section also accepts parallel options (``n_parallel``, or a
``parallel: {mode: parallel_task, num_workers: N}`` block) that dispatch
matrix arms to worker processes — see ``example-configs/multiprocessing-demo.yml``
for a complete example, and :doc:`parallel_execution` for the in-process
executor backend that parallelizes *within* a single run. The two are
orthogonal: process-level parallelism across matrix arms,
executor-level parallelism across clients.

Next steps
----------

- :doc:`cli` — the full YAML syntax reference and CLI options.
- :doc:`parallel_execution` — speed up each individual run.
- :ref:`howto-custom-algorithm`, :ref:`howto-custom-dataset` and
  :ref:`howto-custom-optimizer` — extend the framework itself; the
  ``test-files/custom_*.py`` files they walk through are complete,
  runnable examples that the CLI accepts as algorithm/dataset paths.
