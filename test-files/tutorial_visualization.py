"""
Visualize and export experiment results.

This script is the source of the code blocks shown in the
``tutorial-visualization`` page of the documentation
(``docs/source/tutorial-visualization.rst``): the marked sections are
included verbatim via Sphinx ``literalinclude`` directives, so the docs
and this script can never drift apart. The training runs below are
pinned to the tutorial budget (``num_iters <= 5``, ``num_clients <= 30``,
one local epoch), enforced by ``test/test_tutorial_scripts.py``.

The two runs at the top produce the experiment logs used below; the same
code works on any log directory, including those from the earlier
tutorials.

The interactive part of the tutorial (the ``Panel`` dashboard) needs a
Jupyter notebook and is therefore only constructed, not rendered, here.
"""

# --- begin: setup ---
from fl_sim.algorithms.fedopt import FedAvgClientConfig, FedAvgServer, FedAvgServerConfig
from fl_sim.data_processing.fed_synthetic import FedSynthetic

dataset = FedSynthetic(1.0, 1.0, True, 30)

# two seeds of the same experiment: the minimal multi-seed setup
for seed in (0, 1):
    server = FedAvgServer(
        # a fresh model per run: the server trains the model it is given
        # in place, so sharing one module would warm-start the second run
        dataset.candidate_models["mlp_d1"],
        dataset,
        FedAvgServerConfig(
            num_iters=3,
            num_clients=30,
            clients_sample_ratio=0.5,
            seed=seed,
            log_dir="t3_seeds",
            tag=f"seed_{seed}",  # distinct tag per seed: no log overwrites
        ),
        FedAvgClientConfig(batch_size=32, num_epochs=1),
    )
    server.train()
# --- end: setup ---

from pathlib import Path

# --- begin: plot-mean ---
import matplotlib.pyplot as plt

from fl_sim.utils.viz import aggregate_results_from_json_log_with_custom_axis, find_log_files

# the runs above logged into the t3_seeds directory under fl-sim's log root
log_dir = Path(server._logger_manager.log_dir)
files = find_log_files(directory=log_dir, filters="seed_")
curves = [aggregate_results_from_json_log_with_custom_axis(f, part="val", x_metric="step", y_metric="acc") for f in files]

fig, ax = plt.subplots(figsize=(6, 4))
from fl_sim.utils.viz import plot_mean_curve_with_error_bounds

fig, ax = plot_mean_curve_with_error_bounds(curves, error_type="sem", label="FedAvg", fig_ax=(fig, ax))
ax.set_xlabel("cumulative communications")
ax.set_ylabel("val acc")
fig.savefig("fedavg-multi-seed.pdf", bbox_inches="tight")
# --- end: plot-mean ---

# --- begin: panel ---
from fl_sim.utils.viz import Panel

# in a Jupyter notebook this renders an interactive dashboard:
# pick log files, tune curves, export to PDF/SVG/PNG
panel = Panel(logdir="t3_seeds")
# --- end: panel ---
