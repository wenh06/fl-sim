"""
Minimal federated training run: FedAvg on synthetic data.

This script is the source of the code blocks shown in the Quickstart
page of the documentation (``docs/source/quickstart.rst``): the marked
sections are included verbatim via Sphinx ``literalinclude`` directives,
so the docs and this script cannot drift apart. The size is pinned to
the tutorial budget (``num_iters <= 5``, ``num_clients <= 30``, one local
epoch), enforced by ``test/test_tutorial_scripts.py``.
"""

# --- begin: core ---
from fl_sim.algorithms.fedopt import FedAvgClientConfig, FedAvgServer, FedAvgServerConfig
from fl_sim.data_processing.fed_synthetic import FedSynthetic

# a synthetic dataset split across 30 clients (no download needed)
dataset = FedSynthetic(1.0, 1.0, True, 30)
model = dataset.candidate_models["mlp_d1"]

server = FedAvgServer(
    model,
    dataset,
    FedAvgServerConfig(
        num_iters=5,
        num_clients=10,
        clients_sample_ratio=1.0,
        log_dir="quickstart",  # logs are written to ~/.cache/fl-sim/.logs/quickstart/
    ),
    FedAvgClientConfig(batch_size=32, num_epochs=1),
)
server.train()
# --- end: core ---

# --- begin: read-logs ---
import json
from pathlib import Path

# the JSON metrics logger buffers in memory; flush it to materialize the file
server._logger_manager.flush()

log_dir = Path(server._logger_manager.log_dir)
for log_file in sorted(log_dir.iterdir()):
    print("log file:", log_file.name)

# one json file holds the metrics history, config.json holds the run configs
metrics_file = next(p for p in log_dir.glob("*.json") if p.name != "config.json")
with metrics_file.open() as f:
    metrics = json.load(f)

latest = metrics["val"]["Server"][-1]
print(f"Latest logged val/acc: {latest['acc']:.4f} (step {latest['step']})")
# --- end: read-logs ---
