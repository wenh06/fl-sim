"""
command line interface for experiments

Reads in a yaml file with experiment parameters and runs the experiment.
"""

import argparse
import os
import re
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List, Union

import yaml
from torch_ecg.cfg import CFG
from .data_processing import (
    FedCIFAR100,
    FedEMNIST,
    FedMNIST,
    FedShakespeare,
    FedProxFEMNIST,
    FedProxMNIST,
    FedSynthetic,
    FedProxSent140,
)
from .algorithms import (
    ditto,
    feddr,
    fedopt,
    fedpd,
    fedprox,
    fedsplit,
    ifca,
    pfedmac,
    pfedme,
    proxskip,
    scaffold,
)


# create log directory if it does not exist
# (Path(__file__).parent / ".logs").mkdir(exist_ok=True, parents=True)


def parse_args() -> List[CFG]:
    """Parse command line arguments.

    Returns
    -------
    List[CFG]
        A list of configs read from the config file.

    """
    parser = argparse.ArgumentParser(
        description="LCFL Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Config file (.yml or .yaml file) path",
    )

    args = vars(parser.parse_args())
    config_file_path = Path(args["config_file_path"]).resolve()

    assert config_file_path.exists(), f"Config file {config_file_path} not found"
    assert config_file_path.suffix in [".yml", ".yaml"], (
        f"Config file {config_file_path} must be a .yml or .yaml file, "
        f"but got {config_file_path.suffix}."
    )

    return parse_config_file(config_file_path)


def parse_config_file(config_file_path: Union[str, Path]) -> List[CFG]:
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
    file_content = re.sub("(?:^|\\s+)#.*(\\n|$)", "\n", file_content)

    configs = yaml.safe_load(file_content)

    # set env if specified
    if configs.get("env", None) is not None:
        for k, v in configs["env"].items():
            os.environ[k] = str(v)

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

    return configs


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

    algorithm_pool = CFG(
        {
            "Ditto": {
                "server_config": ditto.DittoServerConfig,
                "client_config": ditto.DittoClientConfig,
                "server": ditto.DittoServer,
            },
            "FedPD": {
                "server_config": fedpd.FedPDServerConfig,
                "client_config": fedpd.FedPDClientConfig,
                "server": fedpd.FedPDServer,
            },
            "FedSplit": {
                "server_config": fedsplit.FedSplitServerConfig,
                "client_config": fedsplit.FedSplitClientConfig,
                "server": fedsplit.FedSplitServer,
            },
            "FedDR": {
                "server_config": feddr.FedDRServerConfig,
                "client_config": feddr.FedDRClientConfig,
                "server": feddr.FedDRServer,
            },
            "IFCA": {
                "server_config": ifca.IFCAServerConfig,
                "client_config": ifca.IFCAClientConfig,
                "server": ifca.IFCAServer,
            },
            "pFedMe": {
                "server_config": pfedme.pFedMeServerConfig,
                "client_config": pfedme.pFedMeClientConfig,
                "server": pfedme.pFedMeServer,
            },
            "pFedMac": {
                "server_config": pfedmac.pFedMacServerConfig,
                "client_config": pfedmac.pFedMacClientConfig,
                "server": pfedmac.pFedMacServer,
            },
            "FedOpt": {
                "server_config": fedopt.FedOptServerConfig,
                "client_config": fedopt.FedOptClientConfig,
                "server": fedopt.FedOptServer,
            },
            "FedAdam": {
                "server_config": fedopt.FedAdamServerConfig,
                "client_config": fedopt.FedAdamClientConfig,
                "server": fedopt.FedAdamServer,
            },
            "FedAdagrad": {
                "server_config": fedopt.FedAdagradServerConfig,
                "client_config": fedopt.FedAdagradClientConfig,
                "server": fedopt.FedAdagradServer,
            },
            "FedYogi": {
                "server_config": fedopt.FedYogiServerConfig,
                "client_config": fedopt.FedYogiClientConfig,
                "server": fedopt.FedYogiServer,
            },
            "FedAvg": {
                "server_config": fedopt.FedAvgServerConfig,
                "client_config": fedopt.FedAvgClientConfig,
                "server": fedopt.FedAvgServer,
            },
            "FedProx": {
                "server_config": fedprox.FedProxServerConfig,
                "client_config": fedprox.FedProxClientConfig,
                "server": fedprox.FedProxServer,
            },
            "ProxSkip": {
                "server_config": proxskip.ProxSkipServerConfig,
                "client_config": proxskip.ProxSkipClientConfig,
                "server": proxskip.ProxSkipServer,
            },
            "SCAFFOLD": {
                "server_config": scaffold.SCAFFOLDServerConfig,
                "client_config": scaffold.SCAFFOLDClientConfig,
                "server": scaffold.SCAFFOLDServer,
            },
        }
    )

    dataset_pool = {
        c.__name__: c
        for c in [
            FedCIFAR100,
            FedEMNIST,
            FedMNIST,
            FedShakespeare,
            FedProxFEMNIST,
            FedProxMNIST,
            FedShakespeare,
            FedProxSent140,
            FedSynthetic,
            # FedRotatedMNIST,
            # FedRotatedCIFAR10,
        ]
    }

    # set random seed
    seed = config.pop("seed", None)  # global seed
    if config.dataset.get("seed", None) is None:
        config.dataset.seed = seed
    if config.algorithm.server.get("seed", None) is None:
        config.algorithm.server.seed = seed
    assert config.dataset.seed is not None and config.algorithm.server.seed is not None

    # dataset and model selection
    ds_cls = dataset_pool[config.dataset.pop("name")]
    ds = ds_cls(**(config.dataset))
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
    server_config_cls = algorithm_pool[config.algorithm.name]["server_config"]
    client_config_cls = algorithm_pool[config.algorithm.name]["client_config"]
    server_config = server_config_cls(**(config.algorithm.server))
    client_config = client_config_cls(**(config.algorithm.client))

    # setup the experiment
    server_cls = algorithm_pool[config.algorithm.name]["server"]
    s = server_cls(
        model,
        ds,
        server_config,
        client_config,
        lazy=False,
    )

    s._logger_manager.log_message(f"Experiment config:\n{config_bak}")

    # s._setup_clients()

    # execute the experiment
    s.train_federated()

    # destroy the experiment
    del s, ds, model


def main():
    configs = parse_args()
    # TODO: run multiple experiments in parallel using Ray, etc.
    for config in configs:
        single_run(config)


if __name__ == "__main__":
    # typical usage:
    # nohup python -u cli.py {{config.yml}} > .logs/cli.log 2>&1 & echo $! > .logs/cli.pid
    # replace {{config.yml}} with the path to your config file

    main()