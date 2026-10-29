"""
Your first controlled experiment: FedAvg vs FedProx vs SCAFFOLD.

This script is the source of the code blocks shown in the
``tutorial-first-experiment`` page of the documentation
(``docs/source/tutorial-first-experiment.rst``): the marked sections are
included verbatim via Sphinx ``literalinclude`` directives, so the docs
and this script cannot drift apart. The size is pinned to the tutorial
budget (``num_iters <= 5``, ``num_clients <= 30``, one local epoch),
enforced by ``test/test_tutorial_scripts.py``.

The dataset is built once, before any config or server: config
constructors re-seed the global RNG when they run, so whatever was built
last would otherwise shift the results. Constructing in the order
dataset -> config -> server keeps the results reproducible.
"""

# --- begin: setup ---
import json
from pathlib import Path

import pandas as pd

from fl_sim.algorithms.fedopt import FedAvgClientConfig, FedAvgServer, FedAvgServerConfig
from fl_sim.algorithms.fedpd import FedPDClientConfig, FedPDServer, FedPDServerConfig
from fl_sim.algorithms.fedprox import FedProxClientConfig, FedProxServer, FedProxServerConfig
from fl_sim.algorithms.scaffold import SCAFFOLDClientConfig, SCAFFOLDServer, SCAFFOLDServerConfig
from fl_sim.data_processing.fed_synthetic import FedSynthetic

ALGORITHMS = {
    "FedAvg": (FedAvgServer, FedAvgServerConfig, FedAvgClientConfig),
    "FedProx": (FedProxServer, FedProxServerConfig, FedProxClientConfig),
    "SCAFFOLD": (SCAFFOLDServer, SCAFFOLDServerConfig, SCAFFOLDClientConfig),
    "FedPD": (FedPDServer, FedPDServerConfig, FedPDClientConfig),
}

# SCAFFOLD's server config requires its global step size, and FedPD's
# upload frequency is 0 < p < 1 (it uploads once every int(1/p) rounds);
# the others use their defaults.
SERVER_KW = {"FedAvg": {}, "FedProx": {}, "SCAFFOLD": {"lr": 0.03}, "FedPD": {"p": 0.3}}

# one synthetic dataset, split across 30 clients, shared by every run below
dataset = FedSynthetic(1.0, 1.0, True, 30)


def run(algorithm, sample_ratio, num_iters, tag, **extra_kw):
    """Train one algorithm and return the server.

    Every run takes a FRESH model from ``dataset.candidate_models``
    (a property building a new module on each access): the server trains
    the model it is given in place, so sharing one model across runs
    would warm-start every run after the first and contaminate the
    comparison. Algorithm-specific server keywords (SCAFFOLD's global
    step size, FedPD's upload frequency ``p``) come from ``SERVER_KW``;
    anything passed as ``extra_kw`` is merged on top.
    """
    server_cls, server_config_cls, client_config_cls = ALGORITHMS[algorithm]
    server_kw = {**SERVER_KW[algorithm], **extra_kw}
    server = server_cls(
        dataset.candidate_models["mlp_d1"],
        dataset,
        server_config_cls(
            num_iters=num_iters,
            num_clients=30,
            clients_sample_ratio=sample_ratio,
            log_dir=f"t1_{tag}",
            **server_kw,
        ),
        client_config_cls(batch_size=32, num_epochs=1),
    )
    server.train()
    return server


def final_accuracy(server):
    """Read the last logged server-side validation accuracy of a run.

    The JSON logger buffers in memory; flush it to materialize the file.
    """
    server._logger_manager.flush()
    log_dir = Path(server._logger_manager.log_dir)
    metrics_file = next(p for p in log_dir.glob("*.json") if p.name != "config.json")
    metrics = json.loads(metrics_file.read_text())
    return metrics["val"]["Server"][-1]["acc"]


# --- end: setup ---

# --- begin: compare ---
results = []
for algorithm in ("FedAvg", "FedProx", "SCAFFOLD"):
    for sample_ratio in (0.5, 1.0):
        server = run(algorithm, sample_ratio, num_iters=3, tag=f"{algorithm}_{sample_ratio}")
        results.append(
            {
                "algorithm": algorithm,
                "sample_ratio": sample_ratio,
                "num_communications": server._num_communications,
                "final_acc": round(final_accuracy(server), 4),
            }
        )

print(pd.DataFrame(results).to_string(index=False))
# --- end: compare ---

# --- begin: fedpd ---
fedavg = run("FedAvg", 0.5, num_iters=5, tag="FedAvg_5it")
fedpd = run("FedPD", 0.5, num_iters=5, tag="FedPD_5it")

# The skipped rounds emit expected RuntimeWarnings on both the server
# and the client side -- see the tutorial page for what they mean.
for name, server in [("FedAvg", fedavg), ("FedPD", fedpd)]:
    sampled = 15  # 30 clients x 0.5
    downlink = sampled * 5  # every round broadcasts to every sampled client
    uplink = server._num_communications - downlink
    print(
        f"{name}: {server._num_communications} communications = "
        f"{downlink} downlink + {uplink} uplink ({uplink // sampled} of 5 rounds uploaded)"
    )
# --- end: fedpd ---
