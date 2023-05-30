"""
Functions for collecting experiment results, visualizing them, etc.
"""

import os
import re
from pathlib import Path
from typing import Union, Sequence, Optional, Tuple, List

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from fl_sim.nodes import Node
from fl_sim.utils.const import LOG_DIR


__all__ = [
    "find_log_files",
    "get_config_from_log",
    "plot_curve",
]


# turn off the progress bar for loading log files
os.environ["FLSIM_VERBOSE"] = "0"


def find_log_files(
    directory: Union[str, Path] = LOG_DIR, filters: str = "", show: bool = False
) -> Union[List[Path], None]:
    """Find log files in the given directory, recursively.

    Parameters
    ----------
    directory : Union[str, pathlib.Path], default fl_sim.utils.const.LOG_DIR
        The directory to search for log files.
    filters : str, default ""
        Filters for the log files.
        Only files fitting the pattern of `filters` will be returned.
    show : bool, default False
        Whether to print the found log files.
        If True, the found log files will be printed and **NOT** returned.

    Returns
    -------
    List[pathlib.Path]
        The list of log files.

    """
    log_files = [
        item
        for item in Path(directory).rglob("*.json")
        if item.is_file() and re.search(filters, item.name)
    ]
    if show:
        for idx, fn in enumerate(log_files):
            print(idx, "---", fn.stem)
    else:
        return log_files


def get_config_from_log(file: Union[str, Path]) -> dict:
    """Get the config from the log file.

    Parameters
    ----------
    file : Union[str, pathlib.Path]
        Path to the log file.

    Returns
    -------
    dict
        The config.

    """
    file = Path(file)
    if not file.exists():
        print("File not found")
        return {}
    if file.suffix == ".json":
        file = file.with_suffix(".txt")
    if not file.exists():
        print("Corresponding text log file not found")
        return {}
    contents = file.read_text().splitlines()
    flag = False
    for idx, line in enumerate(contents):
        if "FLSim - INFO - Experiment config:" in line:
            flag = True
            break
    if flag:
        return eval(contents[idx + 1])
    else:
        print("Config not found")
        return {}


def plot_curve(
    files: Union[str, Path, Sequence[Union[str, Path]]],
    part: str = "val",
    metric: str = "acc",
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
    labels: Union[str, Sequence[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the curve of the given part and metric
    from the given log file(s).

    Parameters
    ----------
    files : Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]]
        The log file(s).
    part : str, default "val"
        The part of the data, e.g., "train", "val", "test", etc.
    metric : str, default "acc"
        The metric to plot, e.g., "acc", "top3_acc", "loss", etc.
    fig_ax : Optional[Tuple[plt.Figure, plt.Axes]], default None
        The figure and axes to plot on.
        If None, a new figure and axes will be created.
    labels : Union[str, Sequence[str]], default None
        The labels for the curves.
        If None, the stem of the log file(s) will be used.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes.

    """
    curves = []
    stems = []
    if isinstance(files, (str, Path)):
        files = [files]
    for file in files:
        curves.append(
            Node.aggregate_results_from_json_log(
                file,
                part=part,
                metric=metric,
            )
        )
        stems.append(Path(file).stem)
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig, ax = fig_ax
    plot_config = dict(marker="*")
    if labels is None:
        labels = stems
    for idx, curve in enumerate(curves):
        plot_config["label"] = labels[idx]
        ax.plot(curve, **plot_config)
    ax.legend(loc="best", fontsize=18)
    ax.set_xlabel("Global Iter.", fontsize=14)
    ax.set_ylabel(f"{part} {metric}", fontsize=14)
    return fig, ax


class Panel:
    """Panel for visualizing experiment results.

    Parameters
    ----------
    logdir : Optional[Union[str, pathlib.Path]], optional
        The directory to search for log files.
        Defaults to `fl_sim.utils.const.LOG_DIR`.

    """

    __name__ = "Panel"

    def __init__(self, logdir: Optional[Union[str, Path]] = None) -> None:
        self._logdir = Path(logdir or LOG_DIR).expanduser().resolve()
        self._log_files = find_log_files()
        self._refresh_button = widgets.Button(
            description="Refresh",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Refresh",
            icon="refresh",  # (FontAwesome names without the `fa-` prefix)
        )
        self._filters_input = widgets.Text(
            value="",
            placeholder="",
            description="File filters:",
            disabled=False,
            layout={"width": "300px"},
        )
        self._num_log_files_label = widgets.Label(
            value=f"Found {len(self.log_files)} log files."
        )
        self._log_files_selector = widgets.SelectMultiple(
            options=list(zip(self.log_files, self._log_files)),
            # description="Select log files:",
            disabled=False,
            layout={"width": "500px", "height": "220px"},
        )
        boxed_log_files_selector = widgets.Box(
            [
                widgets.Label(value="Select log files:"),
                self._log_files_selector,
            ]
        )
        self._refresh_button.on_click(self._on_refresh_button_clicked)

        self._part_input = widgets.Text(
            value="val",
            placeholder="val/train/...",
            description="Part:",
            disabled=False,
            layout={"width": "200px"},
        )
        self._metric_input = widgets.Text(
            value="acc",
            placeholder="acc/loss/...",
            description="Metric:",
            disabled=False,
            layout={"width": "200px"},
        )

        # canvas for displaying the curves
        self._canvas = widgets.Output()

        self._show_button = widgets.Button(
            description="Plot the curves",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Plot the curves",
            icon="line-chart",  # (FontAwesome names without the `fa-` prefix)
        )
        self._show_button.on_click(self._on_show_button_clicked)

        # layout
        self._layout = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self._filters_input,
                        self._refresh_button,
                        self._num_log_files_label,
                    ]
                ),
                # self._log_files_selector,
                boxed_log_files_selector,
                widgets.HBox([self._part_input, self._metric_input]),
                self._show_button,
                self._canvas,
            ]
        )

        display(self._layout)

    def _on_refresh_button_clicked(self, button: widgets.Button) -> None:
        # update the list of log files
        self._log_files = find_log_files(filters=self._filters_input.value)
        # update the options of the selector
        self._log_files_selector.options = list(zip(self.log_files, self._log_files))
        # update the label
        self._num_log_files_label.value = f"Found {len(self.log_files)} log files."

    def _on_show_button_clicked(self, button: widgets.Button) -> None:
        # clear the canvas
        self._canvas.clear_output(wait=True)
        # ensure that log files are selected
        if not self._log_files_selector.value:
            with self._canvas:
                print("No log files selected.")
            return
        # ensure that part and metric are specified
        if not self._part_input.value or not self._metric_input.value:
            with self._canvas:
                print("Please specify part and metric.")
            return
        # plot the curves
        with self._canvas:
            try:
                fig, ax = plot_curve(
                    self._log_files_selector.value,
                    part=self._part_input.value,
                    metric=self._metric_input.value,
                )
                widgets.widgets.interaction.show_inline_matplotlib_plots()
            except KeyError:
                print("Invalid part or metric.")
                return

    @property
    def log_files(self) -> List[str]:
        return [item.stem for item in self._log_files]
