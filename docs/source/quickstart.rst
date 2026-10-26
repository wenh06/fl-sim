.. _quickstart:

Quickstart
==========

This page walks you through your first federated training run with fl-sim:
install the package, train :class:`~fl_sim.algorithms.fedopt.FedAvg` on a
synthetic dataset, and look at the logs it produces. The whole run takes
seconds on a laptop — no dataset download, no GPU required.

If you prefer understanding the concepts first, read :doc:`concepts` and
come back.

Prerequisites
-------------

Install fl-sim following :doc:`install` (``pip install git+https://github.com/wenh06/fl-sim.git``).
Python 3.10+ with PyTorch is all you need.

Train your first model
----------------------

Every federated run in fl-sim is built from three objects: a **dataset**
(the data plus how it is split across clients), a **model** (a PyTorch
module, here taken from the dataset's built-in candidates), and a
**server** (which owns the global model, samples clients, and aggregates).
The server config carries the federated schedule (rounds, client count,
sampling ratio); the client config carries the local training recipe.

.. literalinclude:: ../../test-files/quickstart.py
   :language: python
   :start-after: # --- begin: core ---
   :end-before: # --- end: core ---

Paste this into a script (or a REPL) and run it. ``FedSynthetic`` generates
a synthetic classification dataset split across 30 clients — nothing is
downloaded. ``FedAvgServerConfig(num_iters=5, num_clients=10,
clients_sample_ratio=1.0)`` runs **5 rounds**, each sampling **10 of the 30
clients** with all of them responding. ``server.train()`` runs the full
federated loop; you will see per-round progress and metrics in the console.

.. note::

   ``server.train()`` is ``server.train("federated")``. fl-sim can also run
   ``"centralized"`` and ``"local"`` baselines with the same objects — see
   the :meth:`~fl_sim.nodes.Server.train` API.

Where do the logs go?
---------------------

Every run writes its logs to ``~/.cache/fl-sim/.logs/`` (override the
subdirectory with ``log_dir=`` in the server config, as done here). The
script above reads the metrics back and prints the result:

.. code-block:: text

    log file: FedAvg-FedSynthetic-MLP_10-26_12-52.json
    log file: FedAvg-FedSynthetic-MLP_10-26_12-52.txt
    log file: config.json
    Latest logged val/acc: 0.8767 (step 80)

Three files per run:

- a ``.txt`` log with the human-readable console history;
- a ``.json`` log with the full metrics history, structured as
  ``{"train": {...}, "val": {"Server": [...], "Client0": [...], ...}}``
  — one dict per evaluation, holding ``step``, ``acc``, ``loss``, etc.
  This is the file you will plot when visualizing results;
- ``config.json`` — the exact server and client configs of the run, so
  every experiment can be reproduced later.

.. note::

   The ``step`` in the logs counts **communications**, not rounds: each
   round adds one downlink broadcast plus one uplink reply per sampled
   client (10 sampled clients → step advances by 20 per round here). The
   json snapshot is taken at the start of each round, so the final
   communication count is higher than the last logged step.

(The exact accuracy value may differ across machines and package versions;
on one machine this run ends at 0.8767, up from 0.16 at initialization.)

Next steps
----------

- :doc:`concepts` — what actually happens in each round: sampling,
  broadcast, local updates, aggregation.
- :doc:`cli` — run experiments from the command line with YAML config
  files (a full tutorial on experiment batches is on the way).
- :doc:`parallel_execution` — speed training up with a parallel executor
  backend while keeping results reproducible.
