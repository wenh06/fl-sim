.. _tutorial-first-experiment:

Tutorial: a first controlled experiment
=======================================

:doc:`Quickstart <quickstart>` executes a single training run. A complete
experiment compares configurations under identical conditions and reads
the results back. This tutorial compares three algorithms at two sampling
ratios, then examines what "communication cost" denotes in fl-sim.

Everything runs on :class:`~fl_sim.data_processing.fed_synthetic.FedSynthetic`
— no downloads, seconds per run. The full script shown below lives at
``test-files/tutorial_first_experiment.py`` and is exercised in CI.

Prerequisites: :doc:`quickstart` (the three-object model: dataset,
model, server) and :doc:`concepts` (what a round does).

Set up the comparison
---------------------

Six runs: **FedAvg** (plain averaging), **FedProx** (proximal-regularized
local objectives) and **SCAFFOLD** (control variates) — the classic
gradient, proximal and variance-reduction axes — each at **sampling
ratios 0.5 and 1.0**.

One deliberate choice: the dataset is constructed **once, before any
config or server**. Config constructors re-seed the global RNG as they
are built, so whatever was constructed last would otherwise shift the
sampling stream of everything after it. The order *dataset → config →
server* keeps every run reproducible:

.. literalinclude:: ../../test-files/tutorial_first_experiment.py
   :language: python
   :start-after: # --- begin: setup ---
   :end-before: # --- end: setup ---

``final_accuracy`` reads a run's JSON log — the same file format you saw
in :doc:`quickstart`. SCAFFOLD needs its global step size ``lr`` passed
explicitly (its config has no default for it); FedPD will need a
frequency parameter later. Both are per-algorithm server keywords,
collected in ``SERVER_KW``.

Run and read the results
------------------------

The comparison itself is a loop, and the results land in a
:class:`pandas.DataFrame`:

.. literalinclude:: ../../test-files/tutorial_first_experiment.py
   :language: python
   :start-after: # --- begin: compare ---
   :end-before: # --- end: compare ---

Output on one machine (your numbers may differ slightly by package
version):

.. code-block:: text

    algorithm  sample_ratio  num_communications  final_acc
       FedAvg           0.5                  90     0.8114
       FedAvg           1.0                 180     0.8767
      FedProx           0.5                  90     0.8776
      FedProx           1.0                 180     0.8776
     SCAFFOLD           0.5                  90     0.8767
     SCAFFOLD           1.0                 180     0.8767

Two observations about this table:

- This synthetic task is *easy* — FedProx and SCAFFOLD reach the same
  accuracy ceiling as FedAvg at full sampling. Their advantages appear on
  harder, more heterogeneous tasks (see the
  :doc:`algorithm theory pages <algorithms/overview>` for when each
  helps).
- Accuracy saturates on this task; the **communication** cost does not.
  Every run at ratio 0.5 used exactly 90 communications and every run at
  ratio 1.0 used exactly 180 — regardless of the algorithm. This number
  warrants a closer look, because it is easy to misread.

What ``num_communications`` counts
----------------------------------

``num_communications`` counts **messages, not rounds**: each round adds
one downlink broadcast *per sampled client* plus one uplink reply *per
sampled client*. With 30 clients at ratio 0.5, every round is
15 downlinks + 15 uplinks = 30 messages, so 3 rounds = 90; at ratio 1.0
that doubles to 180.

A discriminating check: **3 rounds at full sampling and 6 rounds at half
sampling both give 180.** The counter scales with
``2 × sampled_clients × rounds``, not with rounds. Reading it as
"communication rounds" would make 90 look like "90 rounds" — off by a
factor of 30.

The decomposition to keep in mind:

.. math::

   \text{FedAvg} = 2 \times \text{sampled} \times \text{rounds}

(Note also that the ``step`` recorded in the JSON log is a snapshot
taken at the *start* of each round: it trails the final counter by the
last round's full cost — read the final count from the server, as the
scripts here do.)

Sampling is not communicating
-----------------------------

All three algorithms above communicate every round. fl-sim also provides
**communication-skipping** algorithms, which modify the
``2 × sampled × rounds`` identity in a way that clarifies its two terms.
Run FedPD — which uploads only every ``int(1/p)``-th round — for five
rounds next to a five-round FedAvg:

.. literalinclude:: ../../test-files/tutorial_first_experiment.py
   :language: python
   :start-after: # --- begin: fedpd ---
   :end-before: # --- end: fedpd ---

Output on one machine:

.. code-block:: text

    FedAvg: 150 communications = 75 downlink + 75 uplink (5 of 5 rounds uploaded)
    FedPD: 90 communications = 75 downlink + 15 uplink (1 of 5 rounds uploaded)

The decomposition is precise: **the downlink is identical** (75 = 5
rounds × 15 clients — broadcasting to the sampled set is never skipped);
FedPD saves *uplinks only*. Five FedPD rounds cost the same 90 messages as
three FedAvg rounds: at equal communication cost, FedPD performs two more
local iterations.

Three practical details for running this example:

- ``p`` must satisfy **0 < p < 1**. With p = 1 FedPD uploads every round
  (degenerating to FedAvg); with p > 1 the upload frequency becomes zero
  and the run crashes. The config exposes ``p`` but the derived upload
  frequency ``int(1/p)`` always wins — passing a ``comm_freq`` yourself
  is silently overwritten.
- **You will see RuntimeWarnings — they are expected, not errors.** The
  server logs ``No message received from the clients...`` once per
  skipped round ("nobody uploaded this round, so the global model did
  not move"), and each skipping client logs ``No parameters received
  from server...`` ("I was not given a new global model this round, so I
  keep working from my current parameters"). At tutorial scale the
  client-side message appears once per (skipped client × skipped round),
  so it accounts for most of the console output. Both are how FedPD is
  designed to behave.
- The JSON ``step`` snapshot trails differently per algorithm: for the
  five-round FedAvg it ends at 120 (the last round's 30 messages are
  unlogged), for FedPD at 75 (its last round never uploads). The final
  counts come from ``server._num_communications``.

Next steps
----------

- :doc:`tutorial-cli-matrix` — run the same kind of comparison from the
  command line, as a YAML-driven experiment matrix.
- :doc:`tutorial-visualization` — merge multi-seed runs of this
  comparison into mean curves with error bounds.
- :doc:`parallel_execution` — speed these runs up with a parallel
  executor without changing the results.
