"""
Functions for collecting experiment results, visualizing them, etc.
"""

import itertools
import os
import re
import warnings
from pathlib import Path
from typing import Union, Sequence, Optional, Tuple, List

import numpy as np
import yaml

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
except (ImportError, ModuleNotFoundError):
    mpl = plt = widgets = display = None
try:
    import seaborn as sns
except (ImportError, ModuleNotFoundError):
    sns = None

from fl_sim.nodes import Node
from fl_sim.utils.const import LOG_DIR


__all__ = [
    "find_log_files",
    "get_config_from_log",
    "get_curves_and_labels_from_log",
    "plot_curves",
]


# turn off the progress bar for loading log files
os.environ["FLSIM_VERBOSE"] = "0"


_linestyle_tuple = [  # name, linestyle (offset, on-off-seq)
    ("solid", (0, ())),
    ("densely dashed", (0, (5, 1))),
    ("densely dotted", (0, (1, 1))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("dashed", (0, (5, 5))),
    ("dotted", (0, (1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("loosely dotted", (0, (1, 10))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
]


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


def get_curves_and_labels_from_log(
    files: Union[str, Path, Sequence[Union[str, Path]]],
    part: str = "val",
    metric: str = "acc",
) -> Tuple[List[np.ndarray], List[str]]:
    """Get the curves and labels (stems) from the given log file(s).

    Parameters
    ----------
    files : Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]]
        The log file(s).
    part : str, default "val"
        The part of the data, e.g., "train", "val", "test", etc.
    metric : str, default "acc"
        The metric to plot, e.g., "acc", "top3_acc", "loss", etc.

    Returns
    -------
    Tuple[List[numpy.ndarray], List[str]]
        The curves and labels.

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
    return curves, stems


def plot_curves(
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
    curves, stems = get_curves_and_labels_from_log(files, part=part, metric=metric)
    marker_cycle = itertools.cycle(("o", "s", "v", "^", "<", ">", "p", "P", "*"))
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    # plot_config = dict(marker="*")
    plot_config = dict()
    if labels is None:
        labels = stems
    for idx, curve in enumerate(curves):
        plot_config["marker"] = next(marker_cycle)
        plot_config["label"] = labels[idx]
        ax.plot(curve, **plot_config)
    ax.legend(loc="best")
    ax.set_xlabel("Global Iter.")
    ax.set_ylabel(f"{part} {metric}")
    return fig, ax


class Panel:
    """Panel for visualizing experiment results.

    Parameters
    ----------
    logdir : Optional[Union[str, pathlib.Path]], optional
        The directory to search for log files.
        Defaults to `fl_sim.utils.const.LOG_DIR`.

    TODO
    ----
    1. add sliders for matplotlib rc params
    2. add a input box and a button for saving the figure
    3. add a box for showing the config of the experiment

    """

    __name__ = "Panel"

    __default_rc_params__ = {
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "legend.title_fontsize": 16,
        "figure.figsize": [12, 6],
    }

    def __init__(
        self,
        logdir: Optional[Union[str, Path]] = None,
        rc_params: Optional[dict] = None,
    ) -> None:
        if widgets is None:
            print(
                "One or more of the required packages is not installed: "
                "ipywidgets, matplotlib."
            )
            return
        self._logdir = Path(logdir or LOG_DIR).expanduser().resolve()
        assert self._logdir.exists(), f"Log directory {self._logdir} does not exist."
        self._rc_params = self.__default_rc_params__.copy()
        self._rc_params.update(rc_params or {})
        assert set(self._rc_params.keys()).issubset(
            set(mpl.rcParams)
        ), f"Invalid rc_params: {set(self._rc_params) - set(mpl.rcParams)}."
        self.reset_matplotlib()
        if sns is not None:
            sns.set()
        else:
            warnings.warn("Seaborn is not installed. One gets better plots with it.")
        for key, value in self._rc_params.items():
            plt.rcParams[key] = value

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
        unit = "files" if len(self._log_files) > 1 else "file"
        self._num_log_files_label = widgets.Label(
            value=f"Found {len(self.log_files)} log {unit}."
        )
        self._log_files_mult_selector = widgets.SelectMultiple(
            options=list(zip(self.log_files, self._log_files)),
            # description="Select log files:",
            disabled=False,
            layout={"width": "500px", "height": "220px"},
        )
        boxed_log_files_selector = widgets.Box(
            [
                widgets.Label(value="Select log files:"),
                self._log_files_mult_selector,
            ]
        )
        self._refresh_button.on_click(self._on_refresh_button_clicked)

        figsize_slider_config = dict(
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        init_fig_width, init_fig_height = self._rc_params["figure.figsize"]
        self._fig_width_slider = widgets.IntSlider(
            value=int(init_fig_width),
            min=6,
            max=20,
            description="Fig. width:",
            **figsize_slider_config,
        )
        self._fig_width_slider.observe(self._on_fig_width_slider_value_changed)
        self._fig_height_slider = widgets.IntSlider(
            value=int(init_fig_height),
            min=3,
            max=15,
            description="Fig. height:",
            **figsize_slider_config,
        )
        self._fig_height_slider.observe(self._on_fig_height_slider_value_changed)
        # TODO: add sliders for changing font sizes:
        # 1. xtick.labelsize
        # 2. ytick.labelsize
        # 3. axes.labelsize
        # 4. legend.fontsize

        slider_box = widgets.GridBox(
            [self._fig_width_slider, self._fig_height_slider],
            layout=widgets.Layout(grid_template_columns="repeat(2, 0.5fr)"),
        )

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
        self._show_fig_flag = False
        self._show_button.on_click(self._on_show_button_clicked)
        self._clear_button = widgets.Button(
            description="Clear",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Clear",
            icon="eraser",  # (FontAwesome names without the `fa-` prefix)
        )
        self._clear_button.on_click(self._on_clear_button_clicked)

        self._log_file_dropdown_selector = widgets.Dropdown(
            options=list(zip(self.log_files, self._log_files)),
            description="Select log file:",
            disabled=False,
            layout={"width": "500px"},
        )
        self._show_config_area = widgets.Output()
        self._log_file_dropdown_selector.observe(
            self._on_log_file_dropdown_selector_change, names="value"
        )

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
                # self._log_files_mult_selector,
                boxed_log_files_selector,
                widgets.HBox([self._part_input, self._metric_input]),
                # slider_box,  # TODO: fix the layout
                widgets.HBox(
                    [self._show_button, self._clear_button],
                    layout=widgets.Layout(align_items="center"),
                ),
                widgets.Box([self._canvas]),
                self._log_file_dropdown_selector,
                self._show_config_area,
            ]
        )

        display(self._layout)

    def _on_refresh_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None:
            return
        # update the list of log files
        self._log_files = find_log_files(filters=self._filters_input.value)
        # update the options of the selector
        self._log_files_mult_selector.options = list(
            zip(self.log_files, self._log_files)
        )
        # update the label
        self._num_log_files_label.value = f"Found {len(self.log_files)} log files."
        # update the dropdown selector
        self._log_file_dropdown_selector.options = list(
            zip(self.log_files, self._log_files)
        )

    def _on_log_file_dropdown_selector_change(self, change: dict) -> None:
        if widgets is None:
            return
        # clear self._show_config_area
        self._show_config_area.clear_output(wait=True)
        # display the config dict
        if not self._log_file_dropdown_selector.value:
            # empty
            return
        with self._show_config_area:
            config = get_config_from_log(self._log_file_dropdown_selector.value)
            if config:
                # print(json.dumps(config, indent=4))
                print(yaml.dump(config, default_flow_style=False))

    def _on_fig_width_slider_value_changed(self, change: dict) -> None:
        if widgets is None:
            return
        self._rc_params["figure.figsize"][0] = change["new"]
        if self._show_fig_flag:
            self._show_fig()

    def _on_fig_height_slider_value_changed(self, change: dict) -> None:
        if widgets is None:
            return
        self._rc_params["figure.figsize"][1] = change["new"]
        if self._show_fig_flag:
            self._show_fig()

    def _on_show_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None:
            return
        self._show_fig()

    def _show_fig(self) -> None:
        # clear the canvas
        self._canvas.clear_output(wait=True)
        self._show_fig_flag = False
        # ensure that log files are selected
        if not self._log_files_mult_selector.value:
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
                self._show_fig_flag = True
                fig, ax = plot_curves(
                    self._log_files_mult_selector.value,
                    part=self._part_input.value,
                    metric=self._metric_input.value,
                )
                widgets.widgets.interaction.show_inline_matplotlib_plots()
            except KeyError:
                print("Invalid part or metric.")

    def _on_clear_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None:
            return
        self._canvas.clear_output(wait=False)
        self._show_fig_flag = False

    @property
    def log_files(self) -> List[str]:
        return [item.stem for item in self._log_files]

    @staticmethod
    def reset_matplotlib() -> None:
        if mpl is None:
            return
        """Reset matplotlib to default settings."""
        mpl.rcParams.update(mpl.rcParamsDefault)
