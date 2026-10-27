"""
Parallelize a run and verify it stays reproducible.

This script is the source of the code blocks shown in the
``tutorial-parallel-reproducible`` page of the documentation
(``docs/source/tutorial-parallel-reproducible.rst``): the marked sections
are included verbatim via Sphinx ``literalinclude`` directives, so the
docs and this script can never drift apart. The runs are pinned to the
tutorial budget (``num_iters <= 5``, ``num_clients <= 30``, one local
epoch), enforced by ``test/test_tutorial_scripts.py``.
"""

# --- begin: setup ---
import torch

from fl_sim.algorithms.fedopt import FedAvgClientConfig, FedAvgServer, FedAvgServerConfig
from fl_sim.data_processing.fed_synthetic import FedSynthetic

dataset = FedSynthetic(1.0, 1.0, True, 30)


def run(executor, tag, rng_mode):
    """One FedAvg run on the given executor backend."""
    return FedAvgServer(
        # a fresh model per run (see the first-experiment tutorial)
        dataset.candidate_models["mlp_d1"],
        dataset,
        FedAvgServerConfig(
            num_iters=3,
            num_clients=30,
            clients_sample_ratio=0.5,
            executor=executor,  # "sequential" (default) or "threads"
            num_workers=2,
            log_dir=f"t4_{tag}",
        ),
        FedAvgClientConfig(batch_size=32, num_epochs=1, rng_mode=rng_mode),
    )


def weights_of(server):
    return list(server.model.state_dict().values())


def bitwise_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(weights_of(a), weights_of(b)))


# --- end: setup ---

# --- begin: per-client-check ---
sequential = run("sequential", "seq", rng_mode="per_client")
threads = run("threads", "threads", rng_mode="per_client")

print(
    f"per-client RNG:  communications equal: "
    f"{sequential._num_communications == threads._num_communications}, "
    f"final weights bitwise equal: {bitwise_equal(sequential, threads)}"
)
# --- end: per-client-check ---

# --- begin: global-check ---
sequential_g = run("sequential", "seq_global", rng_mode="global")
threads_g = run("threads", "threads_global", rng_mode="global")

# with the process-global RNG, the participation sets and the aggregation
# arithmetic stay identical, but the clients' local trajectories consume
# the shared RNG concurrently, so the final weights may differ — this is
# the documented (and expected) statistical-level behaviour
print(
    f"global RNG:      communications equal: "
    f"{sequential_g._num_communications == threads_g._num_communications}, "
    f"final weights bitwise equal: {bitwise_equal(sequential_g, threads_g)} (may be False by design)"
)
# --- end: global-check ---
