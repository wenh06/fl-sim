""" """

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

from fl_sim.utils.viz import Panel

logdir = Path(__file__).parents[1].resolve() / "test-files"


def test_panel():
    p = Panel(logdir=logdir)
    # TODO: test the following:
    # select log files from the SelectMultiple widget and plot curves
    # change the part and the metrics Text widgets and plot curves
    # adjust plt rc params via the slider widgets and plot curves
    # and more....


def test_find_log_files():
    from fl_sim.utils.viz import find_log_files

    files = find_log_files(logdir)
    assert len(files) > 0 and all(f.suffix == ".json" for f in files)
    # filters apply to the file name
    assert find_log_files(logdir, filters="non_existent_pattern") == []
    # show=True prints and returns None
    assert find_log_files(logdir, show=True) is None


def test_get_config_from_log():
    from fl_sim.utils.viz import get_config_from_log

    json_log = list(logdir.glob("*.json"))[0]
    config = get_config_from_log(json_log)
    assert isinstance(config, dict) and len(config) > 0

    # missing files lead to empty dicts (with a message printed)
    assert get_config_from_log(logdir / "non_existent.json") == {}
    assert get_config_from_log(logdir / "no_txt_counterpart.json") == {}


def test_get_curves_and_labels_from_log():
    import numpy as np

    from fl_sim.utils.viz import get_curves_and_labels_from_log

    json_log = list(logdir.glob("*.json"))[0]
    curves, stems = get_curves_and_labels_from_log(json_log, part="train", x_metric="step", y_metric="acc")
    assert len(curves) == len(stems) == 1
    assert set(curves[0].keys()) == {"x", "y"}
    assert len(curves[0]["x"]) == len(curves[0]["y"]) > 0
    assert isinstance(curves[0]["y"], np.ndarray)

    # multiple files at once
    curves, stems = get_curves_and_labels_from_log([json_log, json_log])
    assert len(curves) == len(stems) == 2


def test_plot_curves_and_mean_curve():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fl_sim.utils.viz import get_curves_and_labels_from_log, plot_curves, plot_mean_curve_with_error_bounds

    json_log = list(logdir.glob("*.json"))[0]

    fig, ax = plot_curves(json_log, part="train", x_metric="step", y_metric="acc")
    assert ax.get_ylabel() == "train acc"
    plt.close(fig)

    # mean curve with error bounds, legacy (bare y values) format
    curves, _ = get_curves_and_labels_from_log([json_log, json_log], part="train", x_metric="step", y_metric="acc")
    ys = [c["y"] for c in curves]
    fig, ax = plot_mean_curve_with_error_bounds(ys, error_type="std", label="FedAvg")
    assert ax.get_legend() is not None
    plt.close(fig)

    # custom-axis (dict) format, with ranges
    fig, ax = plot_mean_curve_with_error_bounds(
        curves,
        error_type="quartile",
        x_range=(0, 10**9),
        y_range=(0.0, 1.0),
        show_error_bounds=True,
    )
    plt.close(fig)
