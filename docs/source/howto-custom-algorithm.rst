.. _howto-custom-algorithm:

How to: implement a custom algorithm
====================================

Built-in algorithms cover the classics, but research lives in the
variations. This guide implements a complete algorithm — a FedProx
variant with named parameter exchange and optional variance reduction —
walking through every file you need and every hook you can override.
The running example is ``test-files/custom_alg.py``; the code shown
below is included from that file, so it is exercised in CI (the CLI runs
it via ``test-files/custom_conf.yml``).

The one-sentence model: **an algorithm is two config classes plus hook
overrides on the Server and Client classes** — the training loop, the
communication schedule scaffolding, logging, evaluation and the parallel
executors all stay untouched (:doc:`concepts` explains why).

When do you need this?
----------------------

- your algorithm changes **what clients compute** locally (new local
  objective, extra local state) → override client hooks;
- it changes **what is exchanged or how replies are aggregated** →
  override server hooks;
- it needs **new hyperparameters** → add config fields.

If you only need different *values* for existing hyperparameters, no
code is needed — use the built-in algorithm's config.

The config classes
------------------

Two pydantic models, one for each side. New fields become CLI/CLI-YAML
options automatically; ``server_config_kw_doc`` / ``client_config_kw_doc``
append the inherited fields' docs so the API reference stays complete:

.. literalinclude:: ../../test-files/custom_alg.py
   :language: python
   :start-after: # --- begin: configs ---
   :end-before: # --- end: configs ---

The ``model_validator`` here fixes the local optimizer from the ``vr``
flag — config-level invariants belong in validators, so invalid
combinations fail at construction, not mid-run.

The server hooks
----------------

The server customizes the **downlink** (``communicate``) and the
**aggregation** (``update``). Everything else — sampling, the round
loop, metrics — is inherited:

.. literalinclude:: ../../test-files/custom_alg.py
   :language: python
   :start-after: # --- begin: server ---
   :end-before: # --- end: server ---

Points worth noting:

- ``communicate(target)`` builds a typed :class:`~fl_sim.messages.ServerToClient`
  message; ``extras`` carries algorithm-specific payloads (here,
  server-side gradients for variance reduction) alongside the model.
- ``update()`` runs on the main thread after the sampled clients'
  replies arrive in ``self._received_messages`` — in sampled-client
  order, which is what keeps aggregation bitwise identical across
  :doc:`parallel_execution` backends.
- ``client_cls`` and ``config_cls`` wire the custom classes together;
  ``@register_algorithm()`` (shown above each class) makes the algorithm
  usable by name.

The client hooks
----------------

The client customizes the **local training** (``update``) and the
**uplink** (``communicate``):

.. literalinclude:: ../../test-files/custom_alg.py
   :language: python
   :start-after: # --- begin: client ---
   :end-before: # --- end: client ---

The most important contract on this page is in ``communicate``: it
**builds and returns** a :class:`~fl_sim.messages.ClientToServer` and
touches nothing but the client's own state. The framework sends the
message and maintains the communication counters. That is what makes
the hook safe to run inside a worker thread when
:doc:`parallel execution <parallel_execution>` is on — a hook that
reached into server state would only work sequentially.

Named exchange, by the way
~~~~~~~~~~~~~~~~~~~~~~~~~~

This example exchanges parameters **by name** (a ``model_state``
mapping from parameter name to tensor, i.e. :data:`~fl_sim.messages.ModelState`)
instead of the legacy flat positional list. Names survive reordering
and partial modules, and the loading side (``accept_parameters``,
``update``) looks parameters up by name. The flat-list style still
works everywhere; prefer names for new algorithms.

Register and run
----------------

``@register_algorithm()`` (applied to all four classes above) puts the
algorithm into the registry; from then on the CLI and YAML matrices
accept it like any built-in:

.. code-block:: yaml

    algorithm:
      name: Custom
      server:
        num_iters: 100
        vr: true
      client:
        num_epochs: 5

The same four classes work programmatically, exactly like
:doc:`quickstart`:

.. code-block:: python

    server = CustomServer(
        model, dataset, CustomServerConfig(num_iters=5), CustomClientConfig(num_epochs=1)
    )
    server.train()

How your algorithm is kept honest
---------------------------------

Two layers protect a registered algorithm:

- the **golden regression** suite (``test/test_parallel_equivalence/``)
  pins each algorithm's observable trajectory (messages, decisions,
  counters, metrics positions) so refactors cannot silently change
  behavior;
- the **hook contract test** (``test/test_hook_contract.py``) asserts
  that client hooks touch no server state — the parallel-safety
  property ``communicate`` above was written for.

When you add an algorithm to the repository itself, add it to both.

Legacy hook style
-----------------

Older code overrides ``communicate(self, target)`` on the *client* and
mutates server state directly. It still works under the sequential
backend (with a ``DeprecationWarning``), but it is incompatible with
parallel backends and will be removed; new algorithms should build and
return the typed message as shown above.

Next steps
----------

- :ref:`howto-custom-dataset` — bring your own data partitioning.
- :ref:`howto-custom-optimizer` — add a local optimizer.
- :doc:`parallel_execution` — the full hook contract and backends.
- :doc:`api/nodes` — the complete Server/Client API.
