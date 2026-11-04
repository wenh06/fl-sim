.. _parallel_execution_design:

Design notes: the parallel execution layer
==========================================

.. versionadded:: 0.2.0

This page records the design of the :doc:`parallel execution layer
<parallel_execution>` — the requirements that shaped it, the invariants
that make parallel runs trustworthy, and the reasoning behind the visible
interfaces. It explains *why*; :doc:`parallel_execution` is the practical
guide, and the hook-contract tests (``test/test_hook_contract.py``) are
the executable form of the rules stated here.

Problem and requirements
------------------------

fl-sim simulates every client and the server inside one Python process.
Historically the clients' local steps ran strictly sequentially, so the
wall-clock time of a round is proportional to the number of sampled
clients — the dominant cost once models are non-trivial.

The requirements that shaped the design:

- **Scale**: tens to a few hundred clients, up to about a thousand;
  built-in models range from a few KiB to tens of MiB.
- **Devices**: CPU and GPU both matter.
- **Reproducibility**: statistical-level reproducibility is acceptable
  for parallel runs; bitwise reproducibility stays guaranteed for the
  default sequential path.
- **Dependencies**: heavy dependencies (Ray and friends) may only ever
  enter as optional backends.
- The existing architecture is not a constraint: the main loop, the
  client/server responsibility split and the per-algorithm upload
  protocol were all allowed to change.

Prior work falls into three schools. Process-based simulators isolate
clients in worker processes (strong isolation, serialization cost).
Thread-based shared-memory designs keep clients in one process (cheap
state access, GIL discipline required). Vectorized/functional designs
(batch the whole population with ``vmap``, as in FedJAX) achieve the
best throughput but require rewriting every model and algorithm in
functional form — a different framework. fl-sim takes the second path
with a contract that keeps the third and the first reachable.

Goals and non-goals
-------------------

Goals:

- measurable wall-clock speedup on multi-core CPU and single/multi GPU;
- zero semantic change to the 20 built-in algorithms — algorithms only
  went through one framework-managed protocol migration;
- the default ``sequential`` backend is **bitwise identical** to the
  pre-parallel implementation: communication counters (including the
  "empty" +1 of conditional broadcasts), log ordering, the counting
  semantics of communication-skipping algorithms, and client model
  trajectories;
- a three-level reproducibility promise (below);
- the executor abstraction must not change when new backends arrive;
- no new mandatory dependencies (standard-library
  ``concurrent.futures`` plus what PyTorch/NumPy already provide).

Explicit non-goals, with the reasons:

- **Ray / multi-machine execution** — no distributed use case; the
  interface keeps it reachable as an optional backend;
- **vectorized simulation** — equivalent to rewriting the framework;
- **discrete-event simulation** — orthogonal (a tool for system
  performance evaluation, not for accelerating real computation);
- **bitwise cross-run reproducibility under parallelism** — accepted at
  statistical level by design decision; the sequential path keeps
  bitwise behavior;
- **client memory optimization** — buffering all uploads of a round is
  pre-existing behavior; parallelization neither changes nor optimizes
  it;
- **asynchronous / staleness algorithm semantics** — designed as bulk
  synchronous;
- **experiment-level parallelism** — running many experiments in
  parallel (``fl_sim/utils/multiprocessing.py``) is orthogonal to
  client-level parallelism and intentionally untouched.

Design principles: state-ownership invariants
---------------------------------------------

Two invariants carry the entire design:

**I1 (client-task enclosure)** — code running in the parallel region
may only read and write *its own client's* state plus read-only shared
data; server-side state is untouchable. Enforced mechanically by AST
contract tests: the audited touch-set of every client hook is pinned,
a static name blacklist is a cheap pre-filter and a runtime
``isinstance(v, Server)`` check is the real gatekeeper.

**I2 (server single-threadedness)** — every server-side state change
happens on the main thread, in the fixed sampled-client order within
each round.

Together they yield the **arithmetic-equivalence corollary**: given each
client's inputs (the downlink message, or none) and outputs (the uplink
message, or none), the arithmetic sequence of the aggregation stage is
independent of the execution backend. Backends only change how the
clients' local trajectories are scheduled concurrently — never the
aggregation arithmetic. The threads backend enforces the invariants by
convention, code structure and CI assertions; a process backend would
enforce them by physical isolation. One contract covers both.

The executor abstraction
------------------------

Backends implement a single interface::

    class ClientExecutor(ABC):
        def map(self, fn, items, on_complete=None) -> List[T]: ...

The contract:

- ``map`` applies ``fn`` element-wise and returns results **in the
  order of ``items``** — never in completion order. Consuming results
  in completion order would let the floating-point aggregation order
  drift with the scheduler; progress reporting is carried separately by
  ``on_complete(result, index)``, invoked on the main thread as soon as
  each task finishes.
- task exceptions propagate as the original exception chained with the
  client id (``raise ... from exc``); not-yet-started tasks are
  cancelled best-effort, already-running tasks are allowed to finish
  (documented as-is);
- the object is a context manager: entering saves and applies execution
  resources (the torch intra-op thread budget), exiting restores them.

``map`` instead of futures is deliberate: exposing futures would tempt
callers to move state changes into the concurrent region, breaking I2.

Task decomposition per round
----------------------------

Each round splits into a serial main-thread part (inherently cheap) and
a parallel part (the expensive local steps):

1. **Main thread** — the per-algorithm uplink decision
   (``Server._decide_uplink``; only communication-skipping algorithms
   override it, and the RNG consumption for it moves onto the serial
   main-thread path), then the downlink broadcast
   (``Server.communicate(target)``) with **per-algorithm semantics
   preserved as-is**: APFL broadcasts conditionally on ``n_iter % tau``,
   FedPD on the previous round's participation set, everything else
   unconditionally. "Skips the uplink this round" and "receives no
   downlink this round" are independent mechanisms.
2. **Parallel region** — the per-client task: evaluate (on evaluation
   rounds) → local training → build the reply. The task **always**
   evaluates and trains; only the reply is conditional. Skipping the
   uplink never skips training, evaluation or metric logging —
   otherwise the personalized models of communication-skipping
   algorithms would stop evolving.
3. **Main thread** — replies are consumed in sampled-client order; the
   communication counter follows the per-algorithm semantics
   (ProxSkip counts skipped clients, FedPD does not).

Backends
--------

.. list-table::
   :header-rows: 1

   * - Backend
     - Basis
     - Sweet spot
   * - ``sequential`` (default)
     - the plain loop
     - debugging; bitwise reproducibility
   * - ``threads``
     - ``ThreadPoolExecutor``
     - multi-core CPU and single/multi GPU; PyTorch releases the GIL
       inside compute kernels, so the local steps parallelize despite
       the GIL
   * - ``processes`` (planned)
     - persistent worker pool + client sharding
     - CPU workloads where the GIL bounds the threads backend
   * - ``ray`` (planned, optional dependency)
     - Ray actors
     - multi-machine

Measured speedups for ``threads`` (best of 3; see
``test-files/benchmark_executors.py``): about **1.2x** on small
CPU-bound workloads (GIL contention dominates), **1.4x** and above on
compute-heavier CPU models, and **no benefit on a single GPU** (kernels
from concurrent clients serialize on the device anyway).

Reproducibility ladder
----------------------

Randomness has three consumption channels: torch (dataloader shuffling,
dropout), numpy (client sampling and FedPD decisions) and python
``random`` (defensive). The design moves every *decision-class* RNG
consumption onto the serial main-thread path and defines three levels:

- **L0 (default)** — decisions on the main thread + fixed consumption
  order (I2): aggregation is bitwise identical across backends and the
  participation sets are fixed; the clients' local trajectories are
  reproducible at statistical level only.
- **L1 (opt-in, ``rng_mode="per_client"``)** — each client's train/val
  dataloaders carry per-client ``torch.Generator`` objects seeded
  deterministically from ``(seed, round, client_id)`` and re-armed every
  round, making the shuffle streams bitwise identical across backends,
  parallelism degrees and repeats — including the final model weights.
  Dropout and the evaluation-path shuffling are *not* covered (they
  consume the process-global RNG). A ``fork_rng()``-based alternative
  was measured and rejected: ``manual_seed`` writes global state that
  concurrent tasks clobber.
- **L2 (rejected)** — full functional-RNG isolation: too invasive for a
  marginal benefit.

Configuration surface
---------------------

Three keyword-only fields on the server config — ``executor``,
``num_workers``, ``torch_num_threads`` — plus the environment overrides
``FL_SIM_EXECUTOR`` / ``FL_SIM_NUM_WORKERS``. Because the server config
also accepts positional arguments, the new fields are registered in an
internal skip-list so that adding them cannot shift the positions of
the inherited fields.

Implementation notes
--------------------

Points where the shipped implementation differs from, or sharpened, the
original plan:

- The message protocol settled on **named-parameter exchange**
  (``model_state`` mappings) rather than flat positional parameter
  lists; IFCA (cluster centers) and FedCR (encoder/head split) adopt it
  for their non-standard state layouts.
- The bitwise migration harness (25 scenario variants digesting
  messages, decisions, counters and logs) surfaced a pre-existing APFL
  bug: with ``tau > 1`` the server aggregated on broadcast rounds
  instead of period-end rounds. The fix (aggregate at period end) is
  part of this release; migration preserved behavior bit-for-bit,
  including the bug, before the fix.
- L1 was implemented by rebuilding the dataloaders with per-client
  generators in ``Client.__init__`` — deliberately *not* by touching
  the ten ``get_dataloader`` overrides, which keeps the invasive
  surface small.
- Legacy client hooks of the form ``communicate(self, target)`` keep
  working under the sequential backend through a signature-introspection
  shim (with a ``DeprecationWarning``); under parallel backends they
  raise instead of failing silently.

Outlook
-------

The ``processes`` backend needs no changes to the main loop or the
algorithms: I1 already guarantees that a client task's output
(``build_reply`` result) is picklable, and I2 holds by construction on
the main thread. The remaining work is a persistent worker pool with
client sharding and the per-worker device placement policy. A ``ray``
backend would sit behind the same interface as an optional dependency.

.. seealso::

   :doc:`parallel_execution` — the practical guide (quick start,
   configuration, measured speedups); ``test/test_hook_contract.py`` —
   the executable contract tests.
