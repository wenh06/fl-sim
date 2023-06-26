"""
Functions for collecting experiment results, visualizing them, etc.
"""

import itertools
import os
import re
from pathlib import Path
from typing import Union, Sequence, Optional, Tuple, List, Dict, Any

import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from torch_ecg.utils import MovingAverage

try:
    import ipywidgets as widgets
    from IPython.display import display
except (ImportError, ModuleNotFoundError):
    widgets = display = None

from ..nodes import Node
from .const import LOG_DIR
from .misc import is_notebook, find_longest_common_substring


__all__ = [
    "find_log_files",
    "get_config_from_log",
    "get_curves_and_labels_from_log",
    "plot_curves",
    "plot_mean_curve_with_error_bounds",
    "Panel",
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
# _linestyle_cycle = itertools.cycle([ls for _, ls in _linestyle_tuple])
_marker_cycle = ("o", "s", "v", "^", "<", ">", "p", "P", "*")

# fmt: off
_color_palettes = [
    # seaborn color palettes
    "deep", "muted", "bright", "pastel", "dark", "colorblind",
    # matplotlib color palettes
    "tab10", "tab20", "tab20b", "tab20c",
    "Pastel1", "Pastel2", "Paired", "Accent", "Dark2",
    "Set1", "Set2", "Set3",
]
# fmt: on
sns.set_palette("tab10")

DEFAULT_RC_PARAMS = {
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.labelsize": 22,
    "legend.fontsize": 18,
    "legend.title_fontsize": 20,
    "figure.figsize": [16, 8],
    "lines.linewidth": 2.6,
    "font.family": ["sans-serif"],
}
if mpl is not None:
    # NOTE: to use Windows fonts on a Linux machine (e.g. Ubuntu),
    # one can execute the following commands:
    # sudo apt install ttf-mscorefonts-installer
    # sudo fc-cache -fv
    font_files = mpl.font_manager.findSystemFonts()
    for font_file in font_files:
        try:
            mpl.font_manager.fontManager.addfont(font_file)
        except Exception:
            pass
    _font_names = [item.name for item in mpl.font_manager.fontManager.ttflist]
    _fonts_priority = [
        "Helvetica",  # recommended by science (https://www.science.org/content/page/instructions-preparing-initial-manuscript)
        "Arial",  # alternative to Helvetica for Windows users
        "TeX Gyre Heros",  # alternative to Helvetica for Linux users
        "Roboto",  # alternative to Helvetica for Android users
        "Arimo",  # alternative to Helvetica, cross-platform
        "Nimbus Sans",  # alternative to Helvetica
        "CMU Serif",  # Computer Modern Roman, default for LaTeX, serif font
        "JDLangZhengTi",  # sans-serif font for Chinese
        "Times New Roman",  # less recommended with serif font
        "DejaVu Sans",  # default sans-serif font for matplotlib
    ]
    _fonts_priority = [item for item in _fonts_priority if item in _font_names]
    if len(_fonts_priority) == 0:
        _fonts_priority = ["sans-serif"]
    for font in _fonts_priority:
        if font in _font_names:
            DEFAULT_RC_PARAMS["font.family"] = [font]
            break
    print(f"FL-SIM Panel using default font {DEFAULT_RC_PARAMS['font.family']}")
    mpl.rcParams.update(DEFAULT_RC_PARAMS)
    plt.rcParams.update(DEFAULT_RC_PARAMS)
else:
    _font_names, _fonts_priority = None, None


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
        return sorted(log_files)


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
        print(colored("File not found", "red"))
        return {}
    if file.suffix == ".json":
        file = file.with_suffix(".txt")
    if not file.exists():
        print(colored("Corresponding text log file not found", "red"))
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


def _plot_curves(
    curves: Sequence[np.ndarray],
    labels: Sequence[str],
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the curves.

    Parameters
    ----------
    curves : Sequence[numpy.ndarray]
        The curves.
    labels : Sequence[str]
        The labels.
    fig_ax : Tuple[plt.Figure, plt.Axes]
        The figure and axes to plot on.

    """
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    # plot_config = dict(marker="*")
    linestyle_cycle = itertools.cycle([ls for _, ls in _linestyle_tuple])
    marker_cycle = itertools.cycle(_marker_cycle)
    plot_config = dict()
    for idx, curve in enumerate(curves):
        plot_config["marker"] = next(marker_cycle)
        plot_config["linestyle"] = next(linestyle_cycle)
        plot_config["label"] = labels[idx]
        ax.plot(curve, **plot_config)
    ax.legend(loc="best")
    ax.set_xlabel("Global Iter.")
    return fig, ax


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
    if labels is None:
        labels = stems
    fig, ax = _plot_curves(curves, labels, fig_ax)
    ax.set_ylabel(f"{part} {metric}")
    return fig, ax


def plot_mean_curve_with_error_bounds(
    curves: Sequence[np.ndarray],
    error_type: str = "std",
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
    label: Optional[str] = None,
    error_bound_label: bool = True,
    plot_config: Optional[Dict[str, Any]] = None,
    fill_between_config: Dict[str, Any] = {"alpha": 0.3},
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the mean curve with error bounds.

    Parameters
    ----------
    curves : Sequence[np.ndarray]
        The curves.
    error_type : {"std", "sem", "quartile", "iqr"}, default "std"
        The type of error bounds. Can be one of
            - "std": standard deviation
            - "sem": standard error of the mean
            - "quartile": quartile
            - "iqr": interquartile range
    fig_ax : Optional[Tuple[plt.Figure, plt.Axes]], optional
        The figure and axes to plot on.
        If None, a new figure and axes will be created.
    label : Optional[str], optional
        The label for the mean curve.
        Default to ``"mean"``.
    error_bound_label : bool, default True
        Whether to add the label for the error bounds.
    plot_config : Optional[Dict[str, Any]], optional
        The plot config for the mean curve passed to ``ax.plot``.
    fill_between_config : Dict[str, Any], default {"alpha": 0.3}
        The config for ``ax.fill_between``.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes.

    """
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    # allow curves to have different lengths
    max_len = max([len(curve) for curve in curves])
    for idx, curve in enumerate(curves):
        curves[idx] = np.pad(
            curve, (0, max_len - len(curve)), "constant", constant_values=np.nan
        )
    curves = np.array(curves)
    mean_curve = np.nanmean(curves, axis=0)
    if error_type == "std":
        std_curve = np.nanstd(curves, axis=0)
        upper_curve = mean_curve + std_curve
        lower_curve = mean_curve - std_curve
    elif error_type == "sem":
        std_curve = np.nanstd(curves, axis=0)
        upper_curve = mean_curve + std_curve / np.sqrt(len(curves))
        lower_curve = mean_curve - std_curve / np.sqrt(len(curves))
    elif error_type == "quartile":
        q3 = np.nanquantile(curves, 0.75, axis=0)
        q1 = np.nanquantile(curves, 0.25, axis=0)
        upper_curve = q3
        lower_curve = q1
    elif error_type == "iqr":
        q3 = np.nanquantile(curves, 0.75, axis=0)
        q1 = np.nanquantile(curves, 0.25, axis=0)
        iqr = q3 - q1
        upper_curve = q3 + 1.5 * iqr
        lower_curve = q1 - 1.5 * iqr
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    ax.plot(mean_curve, label=label or "mean", **(plot_config or {}))
    if error_bound_label:
        _error_type = {
            "std": "STD",
            "sem": "SEM",
            "quartile": "Quartile",
            "iqr": "IQR",
        }[error_type]
        fill_between_config["label"] = (
            error_type if label is None else f"{label}±{_error_type}"
        )
    ax.fill_between(
        np.arange(len(mean_curve)),
        lower_curve,
        upper_curve,
        **fill_between_config,
    )
    ax.legend(loc="best")
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
    1. ~~add sliders for matplotlib rc params~~(done)
    2. ~~add a input box and a button for saving the figure~~(done)
    3. ~~add a box for showing the config of the experiment~~(done)
    4. ~~use `ToggleButtons` or `TagsInput` to specify indicators for merging multiple curves~~(done)
    5. add choices (via `Dropdown`) for color palette
    6. ~~add a dropdown selector for the sub-directories of the log directory~~(done)

    """

    __name__ = "Panel"

    __default_rc_params__ = DEFAULT_RC_PARAMS.copy()

    def __init__(
        self,
        logdir: Optional[Union[str, Path]] = None,
        rc_params: Optional[dict] = None,
        debug: bool = False,
    ) -> None:
        if widgets is None:
            print(
                "One or more of the required packages is not installed: "
                "ipywidgets, matplotlib."
            )
            return
        self._is_notebook = is_notebook()
        if not self._is_notebook:
            print(
                "Panel is only supported in Jupyter Notebook (JupyterLab, Colab, SageMaker, etc.)."
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
        sns.set()
        self.reset_matplotlib(rc_params=self._rc_params)
        self._debug = debug

        self._curve_cache = {}

        self._log_files = find_log_files(directory=self._logdir)
        self._subdir_dropdown_selector = widgets.Dropdown(
            options=["./"]
            + [
                d.name
                for d in self._logdir.iterdir()
                if d.is_dir() and len(list(d.glob("*.json"))) > 0
            ],
            value="./",
            description="Sub-directory:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self._subdir_dropdown_selector.observe(
            self._on_subdir_dropdown_change, names="value"
        )
        self._subdir_refresh_button = widgets.Button(
            description="Refresh",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Refresh",
            icon="refresh",  # (FontAwesome names without the `fa-` prefix)
        )
        self._subdir_refresh_button.on_click(self._on_subdir_refresh_button_clicked)

        self._files_refresh_button = widgets.Button(
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
        self._log_files_mult_selector = widgets.SelectMultiple(
            options=list(zip(self.log_files, self._log_files)),
            description="Select log files:",
            disabled=False,
            layout={"width": "800px", "height": "220px"},
            style={"description_width": "initial"},
        )
        unit = "files" if len(self._log_files) > 1 else "file"
        unit_selected = (
            "files" if len(self._log_files_mult_selector.value) > 1 else "file"
        )
        self._num_log_files_label = widgets.Label(
            value=(
                f"Found {len(self.log_files)} log {unit}. "
                f"Selected {len(self._log_files_mult_selector.value)} log {unit_selected}."
            )
        )
        # clear self._fig_curves, self._fig_stems if selected log files change
        self._log_files_mult_selector.observe(
            self._log_files_mult_selector_changed, names="value"
        )
        self._files_refresh_button.on_click(self._on_files_refresh_button_clicked)

        self._fig_curves, self._fig_stems = None, None
        fig_setup_slider_config = dict(
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        init_fig_width, init_fig_height = self._rc_params["figure.figsize"]
        init_x_ticks_font_size = self._rc_params["xtick.labelsize"]
        init_y_ticks_font_size = self._rc_params["ytick.labelsize"]
        init_axes_label_font_size = self._rc_params["axes.labelsize"]
        init_legend_font_size = self._rc_params["legend.fontsize"]
        init_linewidth = self._rc_params["lines.linewidth"]
        self._fig_width_slider = widgets.IntSlider(
            value=int(init_fig_width),
            min=6,
            max=20,
            description="Fig. width:",
            **fig_setup_slider_config,
        )
        self._fig_width_slider.observe(
            self._on_fig_width_slider_value_changed, names="value"
        )
        self._fig_height_slider = widgets.IntSlider(
            value=int(init_fig_height),
            min=3,
            max=12,
            description="Fig. height:",
            **fig_setup_slider_config,
        )
        self._fig_height_slider.observe(
            self._on_fig_height_slider_value_changed, names="value"
        )
        self._x_ticks_font_size_slider = widgets.IntSlider(
            value=int(init_x_ticks_font_size),
            min=6,
            max=32,
            description="X tick font size:",
            **fig_setup_slider_config,
        )
        self._x_ticks_font_size_slider.observe(
            self._on_x_ticks_font_size_slider_value_changed, names="value"
        )
        self._y_ticks_font_size_slider = widgets.IntSlider(
            value=int(init_y_ticks_font_size),
            min=6,
            max=32,
            description="Y tick font size:",
            **fig_setup_slider_config,
        )
        self._y_ticks_font_size_slider.observe(
            self._on_y_ticks_font_size_slider_value_changed, names="value"
        )
        self._axes_label_font_size_slider = widgets.IntSlider(
            value=int(init_axes_label_font_size),
            min=6,
            max=42,
            description="Axes label font size:",
            **fig_setup_slider_config,
        )
        self._axes_label_font_size_slider.observe(
            self._on_axes_label_font_size_slider_value_changed, names="value"
        )
        self._legend_font_size_slider = widgets.IntSlider(
            value=int(init_legend_font_size),
            min=6,
            max=32,
            description="Legend font size:",
            **fig_setup_slider_config,
        )
        self._legend_font_size_slider.observe(
            self._on_legend_font_size_slider_value_changed, names="value"
        )
        self._linewidth_slider = widgets.FloatSlider(
            value=init_linewidth,
            min=0.6,
            max=4.4,
            description="Line width:",
            **{**fig_setup_slider_config, **{"step": 0.1, "readout_format": ".1f"}},
        )
        self._linewidth_slider.observe(
            self._on_linewidth_slider_value_changed, names="value"
        )
        self._fill_between_alpha_slider = widgets.FloatSlider(
            value=0.3,
            min=0.1,
            max=0.9,
            description="Fill between alpha:",
            **{**fig_setup_slider_config, **{"step": 0.01, "readout_format": ".2f"}},
        )
        self._fill_between_alpha_slider.observe(
            self._on_fill_between_alpha_slider_value_changed, names="value"
        )
        self._moving_averager = MovingAverage()
        self._moving_average_slider = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=0.9,
            description="Curve smoothing:",
            **{**fig_setup_slider_config, **{"step": 0.01, "readout_format": ".2f"}},
        )
        self._moving_average_slider.observe(
            self._on_moving_average_slider_value_changed, names="value"
        )

        slider_box = widgets.GridBox(
            [
                self._fig_width_slider,
                self._fig_height_slider,
                self._linewidth_slider,
                self._x_ticks_font_size_slider,
                self._y_ticks_font_size_slider,
                self._axes_label_font_size_slider,
                self._legend_font_size_slider,
                self._fill_between_alpha_slider,
                self._moving_average_slider,
            ],
            layout=widgets.Layout(
                grid_template_columns="repeat(3, 0.5fr)",
                grid_template_rows="repeat(3, 0.5fr)",
                grid_gap="0px 0px",
            ),
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
        self._refresh_part_metric_button = widgets.Button(
            description="Refresh part/metric",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Refresh part/metric",
            icon="refresh",  # (FontAwesome names without the `fa-` prefix)
        )
        self._refresh_part_metric_button.on_click(
            self._on_refresh_part_metric_button_clicked
        )

        self._merge_curve_method_dropdown_selector = widgets.Dropdown(
            options=[
                ("standard deviation", "std"),
                ("standard error of the mean", "sem"),
                ("quartile", "quartile"),
                ("interquartile range", "iqr"),
            ],
            value="std",
            description="Merge error bound type:",
            style={"description_width": "initial"},
        )
        self._merge_curve_method_dropdown_selector.observe(
            self._on_merge_curve_method_dropdown_selector_value_changed, names="value"
        )
        self._merge_curve_with_err_bound_label_checkbox = widgets.Checkbox(
            value=True,
            description="Merge with error bound label",
            style={"description_width": "initial"},
        )
        self._merge_curve_with_err_bound_label_checkbox.observe(
            self._on_merge_curve_with_err_bound_label_checkbox_value_changed,
            names="value",
        )
        if widgets.__version__ >= "8":
            self._merge_curve_tags_input = widgets.TagsInput(
                value=[],
                allow_duplicates=False,
                placeholder="FedAvg, FedProx, etc.",
                # description="Merge tags:",
                # style={"description_width": "initial"},
            )
            self._merge_curve_tags_input.observe(
                self._on_merge_curve_tags_input_value_changed, names="value"
            )
            merge_curve_tags_box = widgets.VBox(
                [
                    widgets.HBox(
                        [
                            self._merge_curve_method_dropdown_selector,
                            widgets.Label("Merge tags:"),
                            self._merge_curve_tags_input,
                        ]
                    ),
                    self._merge_curve_with_err_bound_label_checkbox,
                ],
            )
        else:  # TagsInput was added in ipywidgets 8.x
            self._merge_curve_tags_input = None
            merge_curve_tags_box = widgets.HTML(
                "<span style='color:red'>"
                f"Curve merging is not supported for ipywidgets {widgets.__version__}. "
                "Please upgrade to ipywidgets 8.x or above."
                "</span>"
            )

        # canvas for displaying the curves
        self._canvas = widgets.Output(layout={"border": "2px solid black"})

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

        self._font_dropdown_selector = widgets.Dropdown(
            options=_fonts_priority,
            value=_fonts_priority[0],
            description="Font family:",
            style={"description_width": "initial"},
        )
        self._font_dropdown_selector.observe(
            self._on_font_dropdown_selector_value_changed, names="value"
        )

        self._palette_dropdown_selector = widgets.Dropdown(
            options=_color_palettes,
            value="tab10",
            description="Palette:",
            style={"description_width": "initial"},
        )
        self._palette_dropdown_selector.observe(
            self._on_palette_dropdown_selector_value_changed, names="value"
        )

        self._savefig_dir_input = widgets.Text(
            value="./images",
            description="Save dir:",
            style={"description_width": "initial"},
        )
        self._savefig_filename_input = widgets.Text(
            value="",
            description="Save filename:",
            style={"description_width": "initial"},
            placeholder="only filename, no extension",
        )
        self._savefig_format_dropdown_selector = widgets.Dropdown(
            value="pdf",
            options=["pdf", "svg", "png", "jpg", "ps"],
            description="Save format:",
            style={"description_width": "initial"},
        )
        self._savefig_button = widgets.Button(
            description="Save",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Save",
            icon="save",  # (FontAwesome names without the `fa-` prefix)
        )
        self._savefig_message_area = widgets.Output(
            layout={"border": "2px solid black"}
        )
        self._savefig_button.on_click(self._on_savefig_button_clicked)

        self._log_file_dropdown_selector = widgets.Dropdown(
            options=list(zip(self.log_files, self._log_files)),
            description="Select log file:",
            disabled=False,
            layout={"width": "500px"},
            style={"description_width": "initial"},
        )
        self._show_config_area = widgets.Output(layout={"border": "2px solid black"})
        self._log_file_dropdown_selector.observe(
            self._on_log_file_dropdown_selector_change, names="value"
        )

        # layout
        self._layout = widgets.VBox(
            [
                widgets.HBox(
                    [self._subdir_dropdown_selector, self._subdir_refresh_button]
                ),
                widgets.HBox(
                    [
                        self._filters_input,
                        self._files_refresh_button,
                        self._num_log_files_label,
                    ]
                ),
                self._log_files_mult_selector,
                widgets.HBox(
                    [
                        self._part_input,
                        self._metric_input,
                        self._refresh_part_metric_button,
                    ]
                ),
                slider_box,
                merge_curve_tags_box,
                widgets.HBox(
                    [
                        self._show_button,
                        self._clear_button,
                        self._font_dropdown_selector,
                        self._palette_dropdown_selector,
                    ],
                    layout=widgets.Layout(align_items="center"),
                ),
                widgets.Box([self._canvas]),
                widgets.VBox(
                    [
                        widgets.HBox(
                            [
                                self._savefig_dir_input,
                                self._savefig_filename_input,
                                self._savefig_format_dropdown_selector,
                            ],
                            layout=widgets.Layout(align_items="center"),
                        ),
                        widgets.HBox(
                            [self._savefig_button, self._savefig_message_area],
                            layout=widgets.Layout(align_items="center"),
                        ),
                    ],
                ),
                self._log_file_dropdown_selector,
                self._show_config_area,
            ]
        )

        if self._debug:
            self._debug_message_area = widgets.Output(
                layout={"border": "5px solid red"},
            )
            self._layout.children = self._layout.children + (self._debug_message_area,)

        display(self._layout)

    def _on_subdir_dropdown_change(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if change["type"] != "change" or change["name"] != "value":
            return
        self._log_files = find_log_files(
            directory=self._logdir / self._subdir_dropdown_selector.value,
            filters=self._filters_input.value,
        )
        self._log_files_mult_selector.options = list(
            zip(self.log_files, self._log_files)
        )
        unit = "files" if len(self._log_files) > 1 else "file"
        unit_selected = (
            "files" if len(self._log_files_mult_selector.value) > 1 else "file"
        )
        self._num_log_files_label.value = (
            f"Found {len(self.log_files)} log {unit}. "
            f"Slected {len(self._log_files_mult_selector.value)} log {unit_selected}."
        )
        self._log_file_dropdown_selector.options = list(
            zip(self.log_files, self._log_files)
        )

    def _on_subdir_refresh_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None or not self._is_notebook:
            return
        self._subdir_dropdown_selector.options = ["./"] + [
            d.name
            for d in self._logdir.iterdir()
            if d.is_dir() and len(list(d.glob("*.json"))) > 0
        ]

    def _on_files_refresh_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None or not self._is_notebook:
            return
        # update the list of log files
        self._log_files = find_log_files(
            directory=self._logdir / self._subdir_dropdown_selector.value,
            filters=self._filters_input.value,
        )
        # update the options of the selector
        self._log_files_mult_selector.options = list(
            zip(self.log_files, self._log_files)
        )
        # update the label
        unit = "files" if len(self._log_files) > 1 else "file"
        unit_selected = (
            "files" if len(self._log_files_mult_selector.value) > 1 else "file"
        )
        self._num_log_files_label.value = (
            f"Found {len(self.log_files)} log {unit}. "
            f"Slected {len(self._log_files_mult_selector.value)} log {unit_selected}."
        )
        # update the dropdown selector
        self._log_file_dropdown_selector.options = list(
            zip(self.log_files, self._log_files)
        )
        # clear loaded curves and stems
        self._fig_curves, self._fig_stems = None, None

    def _log_files_mult_selector_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        # clear self._fig_curves and self._fig_stems
        self._fig_curves, self._fig_stems = None, None
        # update the label
        unit = "files" if len(self._log_files) > 1 else "file"
        unit_selected = (
            "files" if len(self._log_files_mult_selector.value) > 1 else "file"
        )
        self._num_log_files_label.value = (
            f"Found {len(self.log_files)} log {unit}. "
            f"Slected {len(self._log_files_mult_selector.value)} log {unit_selected}."
        )

    def _on_log_file_dropdown_selector_change(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
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
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], (int, float)):
            self._rc_params["figure.figsize"][0] = change["new"]
        if self._show_fig_flag:
            self._show_fig()

    def _on_fig_height_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], (int, float)):
            self._rc_params["figure.figsize"][1] = change["new"]
            self.reset_matplotlib(rc_params=self._rc_params)
        if self._show_fig_flag:
            self._show_fig()

    def _on_x_ticks_font_size_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        self._show_config_area.clear_output(wait=False)
        if isinstance(change["new"], (int, float)):
            self._rc_params["xtick.labelsize"] = change["new"]
            self.reset_matplotlib(rc_params=self._rc_params)
        if self._show_fig_flag:
            self._show_fig()

    def _on_y_ticks_font_size_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], (int, float)):
            self._rc_params["ytick.labelsize"] = change["new"]
            self.reset_matplotlib(rc_params=self._rc_params)
        if self._show_fig_flag:
            self._show_fig()

    def _on_axes_label_font_size_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], (int, float)):
            self._rc_params["axes.labelsize"] = change["new"]
            self.reset_matplotlib(rc_params=self._rc_params)
        if self._show_fig_flag:
            self._show_fig()

    def _on_legend_font_size_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], (int, float)):
            self._rc_params["legend.fontsize"] = change["new"]
            self.reset_matplotlib(rc_params=self._rc_params)
        if self._show_fig_flag:
            self._show_fig()

    def _on_linewidth_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], (int, float)):
            self._rc_params["lines.linewidth"] = change["new"]
            self.reset_matplotlib(rc_params=self._rc_params)
        if self._show_fig_flag:
            self._show_fig()

    def _on_fill_between_alpha_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if self._show_fig_flag:
            self._show_fig()

    def _on_moving_average_slider_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if self._show_fig_flag:
            self._show_fig()

    def _on_refresh_part_metric_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None or not self._is_notebook:
            return
        # clear the loaded curves and stems
        self._fig_curves, self._fig_stems = None, None
        if self._show_fig_flag:
            self._show_fig()

    def _on_merge_curve_tags_input_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if self._show_fig_flag:
            self._show_fig()

    def _on_merge_curve_method_dropdown_selector_value_changed(
        self, change: dict
    ) -> None:
        if widgets is None or not self._is_notebook:
            return
        if self._show_fig_flag:
            self._show_fig()

    def _on_merge_curve_with_err_bound_label_checkbox_value_changed(
        self, change: dict
    ) -> None:
        if widgets is None or not self._is_notebook:
            return
        if self._show_fig_flag:
            self._show_fig()

    def _on_font_dropdown_selector_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], str):
            self._rc_params["font.family"] = change["new"]
            self.reset_matplotlib(rc_params=self._rc_params)
        if self._show_fig_flag:
            self._show_fig()

    def _on_palette_dropdown_selector_value_changed(self, change: dict) -> None:
        if widgets is None or not self._is_notebook:
            return
        if isinstance(change["new"], str):
            sns.set_palette(change["new"])
        if self._show_fig_flag:
            self._show_fig()

    def _on_show_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None or not self._is_notebook:
            return
        self._show_fig()

    def _show_fig(self) -> None:
        # clear the canvas
        self._canvas.clear_output(wait=True)
        self._show_fig_flag = False
        # ensure that log files are selected
        if not self._log_files_mult_selector.value:
            with self._canvas:
                print(colored("No log files selected.", "red"))
            return
        # ensure that part and metric are specified
        if not self._part_input.value or not self._metric_input.value:
            with self._canvas:
                print(colored("Please specify part and metric.", "red"))
            return
        # plot the curves
        with self._canvas:
            try:
                self._show_fig_flag = True
                if self._fig_curves is None or self._fig_stems is None:
                    # first, fetch from self._curve_cache
                    indices = []
                    self._fig_curves, self._fig_stems = [], []
                    for idx, item in enumerate(self._log_files_mult_selector.value):
                        key = self.cache_key(
                            self._part_input.value, self._metric_input.value, item
                        )
                        if key in self._curve_cache:
                            self._fig_curves.append(self._curve_cache[key])
                            self._fig_stems.append(Path(item).stem)
                            indices.append(idx)
                    if len(indices) < len(self._log_files_mult_selector.value):
                        # second, fetch from the log files
                        for idx, item in enumerate(self._log_files_mult_selector.value):
                            (
                                new_fig_curves,
                                new_fig_stems,
                            ) = get_curves_and_labels_from_log(
                                [
                                    item
                                    for idx, item in enumerate(
                                        self._log_files_mult_selector.value
                                    )
                                    if idx not in indices
                                ],
                                part=self._part_input.value,
                                metric=self._metric_input.value,
                            )
                        # put the new curves and stems into self._curve_cache
                        for curve, stem in zip(new_fig_curves, new_fig_stems):
                            key = self.cache_key(
                                self._part_input.value, self._metric_input.value, stem
                            )
                            self._curve_cache[key] = curve
                        # update self._fig_curves and self._fig_stems
                        self._fig_curves.extend(new_fig_curves)
                        self._fig_stems.extend(new_fig_stems)
                    # self._fig_curves, self._fig_stems = get_curves_and_labels_from_log(
                    #     self._log_files_mult_selector.value,
                    #     part=self._part_input.value,
                    #     metric=self._metric_input.value,
                    # )
                if self._debug:
                    with self._debug_message_area:
                        print(f"self._fig_stems: {self._fig_stems}")
                self.fig, self.ax = plt.subplots(
                    figsize=self._rc_params["figure.figsize"]
                )
                raw_indices = set(range(len(self._fig_curves)))
                linestyle_cycle = itertools.cycle([ls for _, ls in _linestyle_tuple])
                if self._merge_curve_tags_input is not None:
                    common_substring = find_longest_common_substring(
                        self._fig_stems, min_len=5
                    )
                    _fig_stems = [
                        item.replace(common_substring, "-") for item in self._fig_stems
                    ]
                    for idx, tag in enumerate(self._merge_curve_tags_input.value):
                        indices = [
                            idx
                            for idx, stem in enumerate(_fig_stems)
                            if re.search(tag + "\\-", stem)
                            or re.search("\\-" + tag, stem)
                        ]
                        if len(indices) == 0:
                            continue
                        self.fig, self.ax = plot_mean_curve_with_error_bounds(
                            curves=[
                                self._moving_averager(
                                    self._fig_curves[idx],
                                    weight=self._moving_average_slider.value,
                                )
                                for idx in indices
                            ],
                            error_type=self._merge_curve_method_dropdown_selector.value,
                            fig_ax=(self.fig, self.ax),
                            label=tag,
                            error_bound_label=self._merge_curve_with_err_bound_label_checkbox.value,
                            plot_config={"linestyle": next(linestyle_cycle)},
                            fill_between_config={
                                "alpha": self._fill_between_alpha_slider.value
                            },
                        )
                        self.ax.get_legend().remove()
                        raw_indices = raw_indices - set(indices)
                raw_indices = sorted(raw_indices)
                if len(raw_indices) > 0:
                    self.fig, self.ax = _plot_curves(
                        [
                            self._moving_averager(
                                self._fig_curves[idx],
                                weight=self._moving_average_slider.value,
                            )
                            for idx in raw_indices
                        ],
                        [self._fig_stems[idx] for idx in raw_indices],
                        fig_ax=(self.fig, self.ax),
                    )
                else:
                    self.ax.legend(loc="best")
                self.ax.set_ylabel(
                    f"{self._part_input.value} {self._metric_input.value}"
                )
                self.ax.set_xlabel("Global Iter.")
                # widgets.widgets.interaction.show_inline_matplotlib_plots()
                # show_inline_matplotlib_plots might not work well for older versions of
                # related packages or systems
                plt.show(self.fig)
            except KeyError:
                print(colored("Invalid part or metric.", "red"))

    def _on_clear_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None or not self._is_notebook:
            return
        self._fig_curves, self._fig_stems = None, None
        self._canvas.clear_output(wait=False)
        self._show_fig_flag = False

    def _on_savefig_button_clicked(self, button: widgets.Button) -> None:
        if widgets is None or not self._is_notebook:
            return
        self._savefig_message_area.clear_output(wait=False)
        with self._savefig_message_area:
            if self._fig_curves is None or self._fig_stems is None:
                print(colored("No figure to save.", "red"))
                return
            if not self._savefig_filename_input.value:
                print(colored("Please specify a filename.", "red"))
                return
            save_fig_dir = Path(self._savefig_dir_input.value).expanduser().resolve()
            save_fig_dir.mkdir(parents=True, exist_ok=True)
            save_fig_filename = save_fig_dir / self._savefig_filename_input.value
            save_fig_filename = save_fig_filename.with_suffix(
                f".{self._savefig_format_dropdown_selector.value}"
            )
            if save_fig_filename.exists():
                print(colored(f"File {save_fig_filename} already exists.", "red"))
                return
            self.fig.savefig(
                save_fig_filename,
                dpi=600,
                bbox_inches="tight",
            )
            print(f"Figure saved to {save_fig_filename}")

    def cache_key(self, part: str, metric: str, filename: Union[str, Path]) -> str:
        """Get the cache key for a curve."""
        return f"{part}-{metric}-{Path(filename).stem}"

    @property
    def log_files(self) -> List[str]:
        return [item.stem for item in self._log_files]

    @property
    def log_files_with_common_substring_removed(self) -> List[str]:
        common_substring = find_longest_common_substring(self.log_files, min_len=5)
        return [item.replace(common_substring, "-") for item in self.log_files]

    @staticmethod
    def reset_matplotlib(rc_params: Optional[Dict[str, Any]] = None) -> None:
        """Reset matplotlib to default settings."""
        if rc_params is None:
            rc_params = mpl.rcParamsDefault
        mpl.rcParams.update(rc_params)
