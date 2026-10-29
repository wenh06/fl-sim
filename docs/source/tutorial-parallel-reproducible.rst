.. _tutorial-parallel-reproducible:

Tutorial: parallel and reproducible runs
========================================

Larger experiments take longer to train. fl-sim can execute the clients'
local steps concurrently on a thread pool — with a one-line config change
— and this tutorial shows how to keep the run *reproducible* while doing
so: at the end, two different backends produce **bitwise identical final
model weights**.

The full script lives at ``test-files/tutorial_parallel_check.py`` and
is exercised in CI. Prerequisites: :doc:`tutorial-first-experiment`.

The mechanics behind this are documented in
:doc:`parallel_execution` (backends, guarantees, and how custom
algorithms stay compatible); this page is the hands-on version.

Enabling parallel execution
---------------------------

Add ``executor="threads"`` to the server config. Nothing else changes —
algorithms, datasets, logging and the participation schedule all behave
identically; only the per-client local steps (evaluate → local update →
build upload) run on a thread pool:

.. literalinclude:: ../../test-files/tutorial_parallel_check.py
   :language: python
   :start-after: # --- begin: setup ---
   :end-before: # --- end: setup ---

The helpers capture what we will compare: the final model weights
(``weights_of``) and an exact equality check over them
(``bitwise_equal`` — ``torch.equal``, not ``allclose``).

The two backends produce identical results
------------------------------------------

Run the same experiment on both backends with the client config's
``rng_mode="per_client"`` — each client's dataloaders get their own
generators seeded from ``(seed, client_id, round)`` — and compare:

.. literalinclude:: ../../test-files/tutorial_parallel_check.py
   :language: python
   :start-after: # --- begin: per-client-check ---
   :end-before: # --- end: per-client-check ---

Output:

.. code-block:: text

    per-client RNG:  communications equal: True, final weights bitwise equal: True

The results are bitwise identical — not merely close, but the same
tensors. This holds regardless of how many workers ran or how the
operating system scheduled them.

Behavior of the global RNG mode
-------------------------------

``rng_mode`` defaults to ``"global"``: clients share the process-global
torch RNG. The script runs the same comparison in that mode too:

.. literalinclude:: ../../test-files/tutorial_parallel_check.py
   :language: python
   :start-after: # --- begin: global-check ---
   :end-before: # --- end: global-check ---

Under the global mode the *participation schedule* and the *aggregation
arithmetic* are still backend-independent (sampling decisions and reply
consumption happen on the main thread in fixed order), so the
communication counts always match. The clients' local trajectories,
however, consume the shared RNG concurrently — so their interleaving
depends on thread scheduling, and the final weights may differ between
backends and between repeats:

.. code-block:: text

    global RNG:      communications equal: True, final weights bitwise equal: True (may be False by design)

(On a small run like this one, the tasks often serialize and even the
weights may match; this is incidental rather than guaranteed.)

Practical guidance:

- **exploring**: global mode is acceptable; run-to-run variation is part
  of the measurement (average over seeds, as in
  :doc:`tutorial-visualization`).
- **paper-grade seed alignment**: either stay on ``"sequential"``, or
  set ``rng_mode="per_client"`` — then every backend, every parallelism
  degree and every repeat produce the same trajectory from the same
  seed.

Next steps
----------

- :doc:`parallel_execution` — the full contract: guarantees per backend,
  the ``FL_SIM_EXECUTOR``/``FL_SIM_NUM_WORKERS`` environment overrides,
  measured speedups, and what custom algorithms must (not) touch.
- :doc:`tutorial-visualization` — plot the runs you just made
  reproducible.
