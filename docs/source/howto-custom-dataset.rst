.. _howto-custom-dataset:

How to: add a custom dataset
============================

A federated dataset in fl-sim combines two components: the **raw data**
and a **partitioning scheme** that spreads it across the simulated
clients. This guide covers the complete example in
``test-files/custom_dataset.py`` (a re-implementation of the FedProx
paper's FEMNIST variant, exercised in CI via
``test-files/custom_conf.yml``) and then derives a minimal template
starting from NumPy arrays.

Choose the base class
---------------------

- :class:`~fl_sim.data_processing.fed_dataset.FedVisionDataset` — image
  data with the usual ``TensorDataset`` plumbing; what the example uses;
- :class:`~fl_sim.data_processing.fed_dataset.FedDataset` — the fully
  abstract base, for anything non-vision.

The required interface is small: load data in ``_preload``, hand out
dataloaders per client in ``get_dataloader``, and score predictions in
``evaluate``.

The full example
----------------

.. literalinclude:: ../../test-files/custom_dataset.py
   :language: python
   :start-after: # --- begin: dataset ---
   :end-before: # --- end: preload ---

``_preload`` runs once at construction: resolve the data directory,
declare the defaults the framework queries (client counts, batch size,
the file layout), set the ``criterion``, download if needed, and read
the raw files into memory. fl-sim datasets are **immutable after
construction** — partitioning happens here, once.

.. literalinclude:: ../../test-files/custom_dataset.py
   :language: python
   :start-after: # --- begin: dataloader ---
   :end-before: # --- end: dataloader ---

The contract of ``get_dataloader``: given a client index (or ``None``
for "everything"), return a **pair** of dataloaders ``(train, test)``
with shuffling enabled — the framework re-requests them every round,
and under ``rng_mode="per_client"`` (:doc:`tutorial-parallel-reproducible`)
their shuffling is what gets re-seeded per client.

.. literalinclude:: ../../test-files/custom_dataset.py
   :language: python
   :start-after: # --- begin: evaluate ---
   :end-before: # --- end: evaluate ---

``evaluate`` turns a batch of predictions and labels into the metric
dictionary that lands in the logs (``acc``, ``loss``, ``num_samples``,
... — the same keys :doc:`quickstart` read back).

.. literalinclude:: ../../test-files/custom_dataset.py
   :language: python
   :start-after: # --- begin: mirrors-and-models ---
   :end-before: # --- end: mirrors-and-models ---

The remaining properties are conventions rather than requirements:
``mirrors`` / ``url`` feed the download mechanism (fallback URLs tried
in order — see the :meth:`~fl_sim.data_processing.fed_dataset.FedDataset.mirrors`
API for the protocol), ``candidate_models`` declares the models that
fit this data (the CLI's ``model: name:`` picks from here),
``label_map`` names the classes, ``doi`` carries citations.

The minimal template
--------------------

From NumPy arrays to a usable federated dataset. The template below has
been verified by end-to-end execution; three conventions it encodes are
easy to miss and costly to debug:

- **define your own ``__init__``**: the abstract
  :class:`~fl_sim.data_processing.fed_dataset.FedDataset` base has no
  constructor of its own (the vision subclass supplies one); accept
  ``(datadir, transform, seed, **extra_config)`` and re-seed via
  ``set_seed`` so every constructor re-seeds the global RNG exactly
  like the built-ins do;
- **the framework requests global loaders too**: ``get_dataloader``
  must handle ``client_idx=None`` ("all clients"), not just per-client
  indices;
- **``candidate_models`` should return fl-sim models** (they carry the
  ``.device`` attribute the device-allocation step relies on).

Two ``load_partition_data*`` methods appear required but are a legacy
tff-style bulk-loading API that the training loop never calls — stub
implementations satisfy the abstract interface:

.. code-block:: python

    import numpy as np
    import torch
    import torch.utils.data as torchdata

    from fl_sim.models import nn as mnn
    from fl_sim.utils.misc import set_seed
    from fl_sim.data_processing._register import register_fed_dataset
    from fl_sim.data_processing.fed_dataset import FedDataset


    @register_fed_dataset()
    class MyArrays(FedDataset):
        __name__ = "MyArrays"

        def __init__(self, datadir=None, transform="none", seed=0, **extra_config):
            super().__init__()
            self.datadir = datadir
            self.transform = transform
            self.seed = seed
            set_seed(seed)  # so that every constructor re-seeds, as the built-ins do
            for k, v in extra_config.items():
                setattr(self, k, v)
            self._preload()

        def _preload(self, datadir=None):
            rng = np.random.default_rng(self.seed)
            n_clients = 30
            self._parts = [  # one (features, labels) pair per client
                (
                    rng.normal(size=(100, 20)).astype(np.float32),
                    rng.integers(0, 2, size=100).astype(np.int64),
                )
                for _ in range(n_clients)
            ]
            self.DEFAULT_TRAIN_CLIENTS_NUM = n_clients
            self.DEFAULT_TEST_CLIENTS_NUM = n_clients
            self.DEFAULT_BATCH_SIZE = 32
            self.criterion = torch.nn.CrossEntropyLoss()

        def get_dataloader(self, train_bs=None, test_bs=None, client_idx=None):
            parts = self._parts if client_idx is None else [self._parts[client_idx]]
            x = np.concatenate([p[0] for p in parts])
            y = np.concatenate([p[1] for p in parts])
            ds = torchdata.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
            dl = torchdata.DataLoader(ds, batch_size=train_bs or self.DEFAULT_BATCH_SIZE, shuffle=True)
            return dl, dl  # train and test loaders; a real dataset separates them

        def evaluate(self, probs, truths):
            return {
                "acc": (probs.argmax(1) == truths).float().mean().item(),
                "loss": self.criterion(probs, truths).item(),
                "num_samples": probs.shape[0],
            }

        # legacy tff-style bulk loaders; the training loop never calls them
        def load_partition_data(self, batch_size=None):
            raise NotImplementedError

        def load_partition_data_distributed(self, process_id, batch_size=None):
            raise NotImplementedError

        @property
        def url(self):
            return ""  # nothing to download

        @property
        def candidate_models(self):
            # fl-sim models carry a .device attribute the framework relies on
            return {"mlp": mnn.MLP(dim_in=20, dim_out=2, ndim=0)}

        @property
        def doi(self):
            return []

Usage
-----

With the ``@register_fed_dataset()`` decorator applied, the dataset is
constructible by name once its module has been imported (or via the
module *path* as the dataset ``name`` — see
:class:`~fl_sim.data_processing.fed_data_args.FedDataArgs`), and every
built-in works on it unchanged:

.. code-block:: python

    import my_arrays  # the module above; registers the dataset

    from fl_sim.data_processing import FedDataArgs
    from fl_sim.algorithms.fedopt import FedAvgClientConfig, FedAvgServer, FedAvgServerConfig

    dataset = FedDataArgs(name="MyArrays")._create_fed_dataset()
    server = FedAvgServer(
        dataset.candidate_models["mlp"],
        dataset,
        FedAvgServerConfig(num_iters=5, num_clients=10, clients_sample_ratio=1.0),
        FedAvgClientConfig(num_epochs=1, batch_size=32),
    )
    server.train()  # identical to :doc:`quickstart` from here on

Next steps
----------

- :ref:`howto-custom-algorithm` — the algorithm side of the contract.
- :doc:`api/datasets` — the full FedDataset API reference.
