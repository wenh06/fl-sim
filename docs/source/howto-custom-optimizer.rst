.. _howto-custom-optimizer:

How to: add a custom optimizer
==============================

The local training step is pluggable: fl-sim keeps its own optimizer
registry, and any optimizer registered there can be selected by name in
the client config. This is the shortest of the three guides; the
complete example is 48 lines (``test-files/custom_optimizer.py``,
exercised in CI via ``test-files/custom_conf.yml``).

Where optimizers plug in
------------------------

fl-sim's local optimizers live in :mod:`fl_sim.optimizers` and
subclass :class:`~fl_sim.optimizers.base.ProxSGD` — the proximal SGD
base that supports the extra local-training knobs federated algorithms
need (proximal terms, variance-reduction buffers passed by
``optimizer.step(...)``, per the custom algorithm in
:ref:`howto-custom-algorithm`).

Registration makes the optimizer discoverable by name — the same
pattern as algorithms and datasets:

.. literalinclude:: ../../test-files/custom_optimizer.py
   :language: python
   :start-after: # --- begin: optimizer ---
   :end-before: # --- end: optimizer ---

This example wraps ``FedProxOptimizer``'s behavior under a new name;
a genuinely new optimizer overrides the ``ProxSGD`` step methods
instead of the constructor.

Usage
-----

Registered optimizers are referenced by name in the client config —
programmatically:

.. code-block:: python

    FedAvgClientConfig(optimizer="CustomOptimizer", lr=1e-2)

or in a YAML experiment matrix:

.. code-block:: yaml

    algorithm:
      client:
        optimizer: CustomOptimizer

The name is resolved through :func:`~fl_sim.optimizers.get_optimizer`
at client setup (a trailing ``Optimizer`` suffix is optional); an
unregistered name fails there with ``Optimizer ... is not registered``.

Caveats
-------

- the optimizer is constructed **once per client per run**; anything
  stateful it keeps lives for the whole federated run;
- some algorithms fix the client optimizer in their client-config
  validators (e.g. the FedProx family selects ``FedProx`` /
  ``FedProx_VR`` from its ``vr`` flag) — for those algorithms the
  ``optimizer`` config field is ignored by design;
- the server-side optimizers (FedAdam, FedYogi, … in the FedOpt
  family) are a different mechanism — those are selected through the
  server config's ``optimizer`` field and are not part of this
  registry.

Next steps
----------

- :ref:`howto-custom-algorithm` — when the *algorithm*, not just the
  optimizer, is what's new.
- :doc:`api/optimizers` — the optimizer API reference.
