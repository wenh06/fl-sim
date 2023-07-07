"""
command line interface for experiments

Reads in a yaml file with experiment parameters and runs the experiment.
"""

import argparse
import inspect
import os
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union

import yaml
from torch_ecg.cfg import CFG

from fl_sim.algorithms import list_algorithms, get_algorithm
from fl_sim.data_processing import FedDataArgs
from fl_sim.utils.const import NAME
from fl_sim.utils.imports import load_module_from_file


def parse_args() -> List[CFG]:
    """Parse command line arguments.

    Returns
    -------
    List[CFG]
        A list of configs read from the config file.

    """
    parser = argparse.ArgumentParser(
        description=f"{NAME} Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Config file (.yml or .yaml file) path",
    )

    args = vars(parser.parse_args())
    config_file_path = Path(args["config_file_path"]).expanduser().resolve()

    assert config_file_path.exists(), f"Config file {config_file_path} not found"
    assert config_file_path.suffix in [".yml", ".yaml"], (
        f"Config file {config_file_path} must be a .yml or .yaml file, "
        f"but got {config_file_path.suffix}."
    )

    return parse_config_file(config_file_path)


def parse_config_file(config_file_path: Union[str, Path]) -> Tuple[List[CFG], int]:
    """Parse config file.

    Parameters
    ----------
    config_file_path : Union[str, Path]
        The path to the config file.

    Returns
    -------
    List[CFG]
        A list of configs read from the config file.

    """
    file_content = Path(config_file_path).read_text()

    # remove all comments (starts with #) from file_content
    # file_content = re.sub("(?:^|\\s+)#.*(\\n|$)", "\n", file_content)
    file_content = yaml.dump(yaml.safe_load(file_content))

    configs = yaml.safe_load(file_content)

    # set env if specified
    if configs.get("env", None) is not None:
        for k, v in configs["env"].items():
            os.environ[k] = str(v)

    if "strategy" not in configs or "matrix" not in configs["strategy"]:
        # no matrix specified, run a single experiment
        # replace pattern of the form ${{ xx.xx... }} with corresponding value
        new_file_content = deepcopy(file_content)
        new_config = CFG(yaml.safe_load(new_file_content))
        pattern = re.compile(
            "\\$\\{\\{ (?:\\s+)?(?P<repkey>\\w[\\.\\w]+\\w)(?:\\s+)?\\}\\}"
        )
        matches = re.finditer(pattern, new_file_content)
        for match in matches:
            repkey = match.group("repkey")
            try:
                repval = eval(f"new_config.{repkey}")
            except Exception:
                raise ValueError(f"Invalid key {repkey} in {config_file_path}")
            rep_pattern = re.compile(f"\\$\\{{{{(?:\\s+)?{repkey}(?:\\s+)?}}}}")
            new_file_content = re.sub(
                rep_pattern,
                re.sub("(\\n[\\.]{3})?\\n$", "", yaml.safe_dump(repval)),
                new_file_content,
                count=1,
            )
        new_config = CFG(yaml.safe_load(new_file_content))
        new_config.pop("strategy", None)
        if "strategy" in configs and configs["strategy"].get("n_parallel", 1) != 1:
            warnings.warn(
                "`n_parallel` is not supported for single experiment, "
                "ignoring `n_parallel`"
            )
        configs = [new_config]
        n_parallel = 1
        return configs, n_parallel

    # number of parallel runs
    # currently not implemented, but kept for future use
    n_parallel = int(configs["strategy"].get("n_parallel", 1))

    # further process configs to a list of configs
    # by replacing values of the pattern ${{ matrix.key }} with the value of key
    # specified by configs["strategy"]["matrix"][key]
    strategy_matrix = OrderedDict(configs["strategy"]["matrix"])
    repalcements = list(product(*strategy_matrix.values()))
    keys = list(strategy_matrix.keys())
    configs = []
    for rep in repalcements:
        new_file_content = deepcopy(file_content)
        for k, v in zip(keys, rep):
            # replace all patterns of the form ${{ matrix.k }} in file_content with v
            # pattern = re.compile(f"\${{{{ matrix.{k} }}}}")
            # allow for arbitrary number (can be 0) of spaces around matrix.k
            pattern = re.compile(f"\\${{{{(?:\\s+)?matrix.{k}(?:\\s+)?}}}}")
            new_file_content = re.sub(
                pattern,
                re.sub("(\\n[\\.]{3})?\\n$", "", yaml.safe_dump(v)),
                new_file_content,
            )
        new_config = CFG(yaml.safe_load(new_file_content))
        new_config.pop("strategy")
        # replace pattern of the form ${{ xx.xx... }} with corresponding value
        pattern = re.compile(
            "\\$\\{\\{ (?:\\s+)?(?P<repkey>\\w[\\.\\w]+\\w)(?:\\s+)?\\}\\}"
        )
        matches = re.finditer(pattern, new_file_content)
        for match in matches:
            repkey = match.group("repkey")
            try:
                repval = eval(f"new_config.{repkey}")
            except Exception:
                raise ValueError(f"Invalid key {repkey} in {config_file_path}")
            rep_pattern = re.compile(f"\\$\\{{{{(?:\\s+)?{repkey}(?:\\s+)?}}}}")
            new_file_content = re.sub(
                rep_pattern,
                re.sub("(\\n[\\.]{3})?\\n$", "", yaml.safe_dump(repval)),
                new_file_content,
                count=1,
            )
        new_config = CFG(yaml.safe_load(new_file_content))
        new_config.pop("strategy")
        configs.append(new_config)

    return configs, n_parallel


def single_run(config: CFG) -> None:
    """run a single experiment.

    Parameters
    ----------
    config : CFG
        The config of the experiment.

    Returns
    -------
    None

    """
    config = CFG(config)
    config_bak = deepcopy(config)

    # set random seed
    seed = config.pop("seed", None)  # global seed
    if config.dataset.get("seed", None) is None:
        config.dataset.seed = seed
    if config.algorithm.server.get("seed", None) is None:
        config.algorithm.server.seed = seed
    assert config.dataset.seed is not None and config.algorithm.server.seed is not None

    # dataset and model selection
    ds = FedDataArgs._create_fed_dataset_from_args(config.dataset)
    model = ds.candidate_models[config.model.pop("name")]

    # fill default values
    if (
        "batch_size" not in config.algorithm.client
        or config.algorithm.client.batch_size is None
    ):
        config.algorithm.client.batch_size = ds.DEFAULT_BATCH_SIZE
    if (
        "num_clients" not in config.algorithm.server
        or config.algorithm.server.num_clients is None
    ):
        config.algorithm.server.num_clients = ds.DEFAULT_TRAIN_CLIENTS_NUM

    # server and client configs
    builtin_algorithms = list_algorithms().copy()
    if config.algorithm.name not in builtin_algorithms:
        algorithm_file = Path(config.algorithm.name).expanduser().resolve()
        if algorithm_file.suffix == ".py":
            # is a .py file
            # in this case, there should be only one algorithm in the file
            algorithm_name = None
        else:
            # of the form /path/to/algorithm_file_stem.algorithm_name
            # in this case, there could be multiple algorithms in the file
            algorithm_file, algorithm_name = str(algorithm_file).rsplit(".", 1)
            algorithm_file = Path(algorithm_file + ".py").expanduser().resolve()
        assert algorithm_file.exists(), (
            f"Algorithm {config.algorithm.name} not found. "
            "Please check if the algorithm file exists and is a .py file, "
            "or of the form ``/path/to/algorithm_file_stem.algorithm_name``"
        )
        algorithm_module = load_module_from_file(algorithm_file)
        # the custom algorithm should be added to the algorithm pool
        # using the decorator @register_algorithm
        new_algorithms = [
            item for item in list_algorithms() if item not in builtin_algorithms
        ]
        if algorithm_name is None:
            # only one algorithm in `new_algorithms` after `load_module_from_file`
            if len(new_algorithms) == 0:
                raise ValueError(
                    f"No algorithm found in {algorithm_file}. "
                    "Please check if the algorithm is registered using "
                    "the decorator ``@register_algorithm`` from ``fl_sim.algorithms``"
                )
            elif len(new_algorithms) > 1:
                raise ValueError(
                    f"Multiple algorithms found in {algorithm_file}. "
                    "Please split the algorithms into different files, "
                    "or pass the algorithm name in the form "
                    "``/path/to/algorithm_file_stem.algorithm_name``"
                )
            algorithm_name = new_algorithms[0]
        else:
            if algorithm_name not in new_algorithms:
                raise ValueError(
                    f"Algorithm {algorithm_name} not found in {algorithm_file}. "
                    "Please check if the algorithm is registered using "
                    "the decorator ``@register_algorithm`` from ``fl_sim.algorithms``"
                )
    else:
        algorithm_name = config.algorithm.name  # builtin algorithm
    algorithm_dict = get_algorithm(algorithm_name)
    server_config_cls = algorithm_dict["server_config"]
    client_config_cls = algorithm_dict["client_config"]
    server_config = server_config_cls(**(config.algorithm.server))
    client_config = client_config_cls(**(config.algorithm.client))

    # setup the experiment
    server_cls = algorithm_dict["server"]

    server_init_kwargs = {}
    if "lazy" in inspect.getfullargspec(server_cls).args:
        server_init_kwargs["lazy"] = False

    s = server_cls(
        model,
        ds,
        server_config,
        client_config,
        **server_init_kwargs,
    )

    s._logger_manager.log_message(f"Experiment config:\n{config_bak}")

    # s._setup_clients()

    # execute the experiment
    s.train_federated()

    # destroy the experiment
    del s, ds, model


def main():
    configs, n_parallel = parse_args()
    # TODO: run multiple experiments in parallel using Ray, etc.
    for config in configs:
        single_run(config)


if __name__ == "__main__":
    main()
