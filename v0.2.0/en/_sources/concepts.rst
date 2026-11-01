.. _concepts:

Core concepts
=============

This page summarizes the four building blocks of every experiment and the
sequence of events within one round of federated training. For a hands-on
introduction, start with :doc:`quickstart`.

Four building blocks
----------------------

FedDataset — the data and how it is split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~fl_sim.data_processing.fed_dataset.FedDataset` bundles a raw
dataset with a **partitioning scheme**: it decides how the data is spread
across the simulated clients and provides each client with its own
dataloaders through ``get_dataloader(client_idx)``. Built-in datasets (FEMNIST,
CIFAR10, …) download on first use; :class:`~fl_sim.data_processing.fed_synthetic.FedSynthetic`
generates synthetic data locally, so the tutorials in this documentation
complete in seconds. Each dataset also declares ``candidate_models`` — the models it
is meant to be paired with (``dataset.candidate_models["mlp_d1"]``, as in
the quickstart). Datasets are **immutable during training**: partitioning
happens once, up front. Custom datasets are covered in a how-to guide.

Model — a plain PyTorch module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models are ordinary :class:`torch.nn.Module` objects; fl-sim does not
wrap them. Any module from ``candidate_models`` — or any user-provided
module — can be used. The server holds the *global* copy; every sampled client gets its
own copy each round, trains it on its local data slice, and sends the
result back.

Server — the orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~fl_sim.nodes.Server` owns the global model and the
federated schedule (how many rounds, how many clients, how they are
sampled), and drives the training loop described below. It never touches
client-local data. Algorithm-specific behavior lives in subclasses:
:class:`~fl_sim.algorithms.fedopt.FedAvgServer`,
:class:`~fl_sim.algorithms.fedprox.FedProxServer`, and 15+ others — each
customizes a handful of well-defined hooks:

- ``communicate(target)`` — the **downlink**: what the server sends to a
  sampled client (typically the current global model);
- ``update()`` — the **aggregation**: how client replies are turned into
  the new global model.

Client — one participant's local state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each :class:`~fl_sim.nodes.Client` holds its own model copy, its data
slice, and its training state. The client-side hooks correspond to the
server's:

- ``update()`` — the **local training**: fit the received model on local
  data (local epochs, local optimizer — all from the client config);
- ``communicate()`` — the **uplink**: build the message sent back to the
  server. Messages are typed objects (:class:`~fl_sim.messages.ClientToServer`)
  carrying the updated model state plus algorithm-specific extras.

Algorithms customize the hooks, not the loop
--------------------------------------------

fl-sim ships 20 registered algorithms — FedAvg, FedProx, SCAFFOLD,
FedPD, ProxSkip, Ditto, … — and they all run in the *same* loop. An
algorithm is only a pair of config classes plus hook overrides: FedProx
changes the client's local objective; SCAFFOLD adds control variates to
the uplink; FedPD lets clients *skip* some uplinks entirely. The
framework calls the hooks in a fixed order (below), so algorithms remain
independent of the infrastructure — logging, evaluation, reproducibility,
parallel execution.

Registering your own makes it usable from the CLI and experiment
matrices by name:

.. code-block:: python

    from fl_sim.algorithms._register import register_algorithm

    @register_algorithm()
    class MyServer(FedAvgServer):
        ...

.. seealso::

   The how-to guide on custom algorithms walks through a complete,
   runnable example; :doc:`parallel_execution` documents the hook
   contract that keeps custom algorithms compatible with every executor.

One round of training
---------------------

``server.train()`` runs ``num_iters`` rounds. Every round proceeds in a
fixed order:

1. **Sampling** — the server picks this round's participants (the
   ``clients_sample_ratio`` fraction of ``num_clients``).
2. **Uplink decision** — per algorithm: most algorithms have every
   sampled client respond; communication-skipping algorithms (FedPD,
   ProxSkip) decide *now*, on the main thread, which clients will upload
   this round.
3. **Downlink broadcast** — the server sends the global model to each
   sampled client (``server.communicate(target)``).
4. **Local steps** — each selected client runs evaluate → local training
   → build the uplink message. Clients never see each other or the
   global state; they touch only their own slice.
5. **Uplink + aggregation** — replies are consumed **in sampled-client
   order** and the algorithm's ``update()`` produces the new global
   model. Metrics (server-side and per-client) are logged here.
6. **Repeat** — for ``num_iters`` rounds; logs and metrics land in
   ``~/.cache/fl-sim/.logs/`` (see :doc:`quickstart`).

Because steps 1–2 and 5 happen on the main thread in fixed order, the
*aggregation arithmetic* is identical no matter how the local steps of
step 4 are executed — sequentially, or concurrently on a thread pool
(:doc:`parallel_execution`). This invariant underlies the reproducibility
guarantees of parallel execution.

The four objects at a glance
----------------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Configured by
     - Responsible for
     - Does not access
   * - :class:`~fl_sim.data_processing.fed_dataset.FedDataset`
     - constructor args
     - raw data, partitioning, dataloaders
     - training state
   * - :class:`~fl_sim.nodes.Client`
     - ``ClientConfig`` subclass
     - its own data slice and model copy
     - other clients, the global model's evolution
   * - :class:`~fl_sim.nodes.Server`
     - ``ServerConfig`` subclass
     - global model, sampling, aggregation, logging
     - client-local data

Where to go next
----------------

- :doc:`quickstart` — run all of this in five rounds on synthetic data.
- :doc:`api/nodes` — the full Server/Client API reference.
- :ref:`config reference <config_reference>` — every config field
  of every algorithm.
- :doc:`viz` — interactive exploration of the produced logs.
