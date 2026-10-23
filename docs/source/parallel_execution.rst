.. _parallel_execution:

Parallel execution layer
========================

.. versionadded:: 0.2.0

fl-sim simulates all clients and the server inside one Python process.
Historically the clients' local training ran strictly sequentially; the
**parallel execution layer** lets the per-client local steps (evaluate →
local update → build the upload) run concurrently, on a pluggable
**executor backend**, while every server-side state change stays on the
main thread in the fixed sampled-client order.

Quick start
-----------

Enable the thread-pool backend with two keyword arguments on the server
config:

.. code-block:: python

    from fl_sim.algorithms.fedopt import FedAvgServerConfig

    config = FedAvgServerConfig(
        num_iters=100,
        num_clients=200,
        clients_sample_ratio=0.1,
        executor="threads",   # default: "sequential" (exactly the pre-parallel behavior)
        num_workers=8,        # 0 = auto (usable CPU count, container-aware)
    )

Nothing else changes: algorithms, datasets and logging behave identically;
the clients' local steps are simply dispatched to a thread pool. The
backends are also selectable per experiment through the environment
variables ``FL_SIM_EXECUTOR`` / ``FL_SIM_NUM_WORKERS`` (which override the
config; used by CI to run the per-algorithm test functions under a parallel
backend).

Backends
--------

.. list-table::
   :header-rows: 1

   * - Backend
     - Config value
     - Notes
   * - Sequential
     - ``"sequential"`` (default)
     - Exactly the pre-parallel behavior, down to log ordering and exception timing.
   * - Threads
     - ``"threads"``
     - :class:`~concurrent.futures.ThreadPoolExecutor`-based. Client state needs no
       serialization (models and datasets stay put); PyTorch releases the GIL inside
       its compute kernels and CUDA calls, so the local-training workload parallelizes
       despite the GIL. The torch intra-op thread budget is automatically reduced to
       ``cores // num_workers`` while active and restored afterwards;
       ``torch_num_threads`` overrides this explicitly.

Measured speedups (12-core CPU, FedAvg on FedSynthetic, 64 clients, 3
iterations, best of 3 runs; see ``test-files/benchmark_executors.py`` to
reproduce): about **1.2×** at 2–4 workers on this tiny-model workload
(MLP, 1–8 local epochs; single-threaded local steps take only ~15–140 ms),
degrading slightly beyond that as GIL contention dominates. For tiny
models the Python glue code (batch loops, optimizer stepping) holds the
GIL most of the time, so the honest ceiling is near 1.2× — parallel
execution pays off with **compute-heavier local training** (larger models,
more/bigger batches, GPU offload) or the (planned) process-based backend.
The per-backend arithmetic guarantees (fixed aggregation order, identical
participation sets) hold regardless of the achieved speedup.

How it works
------------

Each communication round runs as:

1. **Main thread** — the uplink decision (per algorithm, e.g. FedPD's
   stochastic participation) and the downlink broadcast;
2. **Executor** — the per-client task ``evaluate → update → build_reply``,
   which only ever touches *its own* client's state;
3. **Main thread** — the replies are consumed **in the sampled-client
   order**, then metrics aggregation and the global update.

Because step 3 always happens in a fixed order regardless of the backend,
the floating-point aggregation arithmetic is *identical* across backends —
only the clients' local trajectories can interleave.

Reproducibility
---------------

Three levels, spelled out explicitly:

.. list-table::
   :header-rows: 1

   * - Backend
     - What is guaranteed
   * - ``sequential``
     - The full run is bitwise identical to the pre-parallel implementation
       (same machine/version/seed), including communication counters and logs.
   * - ``threads``
     - The **aggregation arithmetic** is bitwise identical to ``sequential``
       (fixed consumption order); the **participation sets** are identical
       (decisions are made on the main thread, serially); only the clients'
       *local trajectories* may differ statistically from the sequential run
       (concurrent consumption of the process-global RNG).
   * - any parallel backend
     - For paper-grade seed alignment, either use ``sequential``, or set
       ``rng_mode="per_client"`` (below).

``rng_mode="per_client"``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.2.0

Setting ``rng_mode="per_client"`` on the **client** config re-arms every
client's train/val dataloaders with per-client :class:`torch.Generator`s
seeded deterministically from ``(seed, client_id, round)``. The shuffle
stream of each client is then bitwise identical across backends, parallelism
degrees, and same-seed repeats — including the final model weights.

.. code-block:: python

    from fl_sim.algorithms.fedopt import FedAvgClientConfig

    client_config = FedAvgClientConfig(32, 2, lr=0.01, rng_mode="per_client")

Scope: **only the dataloader shuffling** is isolated this way; dropout and
other model-internal randomness still draw from the process-global torch
RNG (isolating those would require functional-RNG rewrites; the practical
recipe for bitwise comparisons remains ``sequential``).

Writing custom algorithms
-------------------------

A client algorithm hook that wants to be parallel-safe obeys one rule:

- ``Client.communicate(self)`` **builds and returns** the
  :class:`~fl_sim.messages.ClientToServer` message and never touches the
  server (the upload and the communication counter are handled by the
  framework);
- legacy-style overrides ``communicate(self, target)`` (uploading through
  ``target`` themselves) still work on the ``sequential`` backend (with a
  ``DeprecationWarning``) but are **rejected** on parallel backends, since
  reading server state from a worker task is impossible by construction.

All built-in algorithms follow the new contract; see
``test-files/custom_alg.py`` for a complete example.

Relation to the experiment-level multiprocessing
------------------------------------------------

:mod:`fl_sim.utils.multiprocessing` (and the ``Server._subprocess`` progress
reporting) provides *experiment-level* parallelism: several experiment
configs → several spawned processes. The executor layer provides
*client-level* parallelism *within* one experiment. The two are orthogonal
and can be combined; the progress reporting works unchanged under both.

API
---

.. automodule:: fl_sim.executors
   :members:
