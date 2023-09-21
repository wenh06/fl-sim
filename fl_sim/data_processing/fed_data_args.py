from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from ..utils.imports import load_module_from_file
from ._register import get_fed_dataset, list_fed_dataset
from .fed_dataset import FedDataset

__all__ = [
    "FedDataArgs",
]


@dataclass
class FedDataArgs:
    """Arguments for creating a federated dataset.

    Parameters
    ----------
    name : str
        name of built-in dataset, or path to the custom dataset file (.py file),
        or path to the custom dataset class (/path/to/py-filename.class_name)
    datadir : str, optional
        path to dataset storage directory, by default None
    transform : str, optional
        name of transform, by default None
    seed : int, optional
        random seed for dataset, by default 0

    Examples
    --------
    .. code-block:: python

        from fl_sim.data_processing import FedDataArgs
        # built-in dataset
        args = FedDataArgs(name="FedMNIST")
        # custom dataset
        args = FedDataArgs(name="/path/to/custom/dataset.py")
        ds = args._create_fed_dataset()
        # another form of custom dataset
        args = FedDataArgs(name="/path/to/custom/dataset.FedCustomDataset")
        ds = args._create_fed_dataset()

    """

    __name__ = "FedDataArgs"

    name: str
    datadir: Optional[str] = None
    transform: Optional[str] = None
    seed: int = 0
    # additional keyword arguments
    # for potential updates and custom federated dataset
    kwargs: Optional[Dict] = None

    def _create_fed_dataset(self) -> FedDataset:
        """Create a federated dataset.

        Returns
        -------
        FedDataset
            a federated dataset

        Examples
        --------
        .. code-block:: python

            from fl_sim.data_processing import FedDataArgs
            # built-in dataset
            args = FedDataArgs(name="FedMNIST")
            # custom dataset
            args = FedDataArgs(name="/path/to/custom/dataset.py")
            ds = args._create_fed_dataset()
            # another form of custom dataset
            args = FedDataArgs(name="/path/to/custom/dataset.FedCustomDataset")
            ds = args._create_fed_dataset()

        """
        return self._create_fed_dataset_from_args(asdict(self))

    @classmethod
    def _add_parser_args(cls, parser: ArgumentParser) -> ArgumentParser:
        default_obj = cls()
        ds_group = parser.add_argument_group()
        ds_group.add_argument(
            "--name",
            type=str,
            required=True,
            help=(
                "name of built-in dataset, "
                "or path to the custom dataset file (.py file), "
                "or path to the custom dataset class (/path/to/py-filename.class_name)"
            ),
        )
        ds_group.add_argument(
            "--datadir",
            type=str,
            default=None,
            help="path to dataset storage directory",
        )
        ds_group.add_argument(
            "--transform",
            type=str,
            default=None,
            help="name of transform",
        )
        ds_group.add_argument(
            "--seed",
            type=int,
            default=0,
            help="random seed for dataset",
        )
        return parser

    @classmethod
    def _create_fed_dataset_from_args(cls, args: Dict) -> FedDataset:
        fed_dataset_name = args.pop("name")
        # if args["name"] is a path to a dataset file, import the dataset from the file
        builtin_fed_dataset_names = list_fed_dataset().copy()
        if fed_dataset_name in builtin_fed_dataset_names:
            fed_dataset_cls = get_fed_dataset(fed_dataset_name)
        else:
            if fed_dataset_name.endswith(".py"):
                # is a .py file
                # in this case, there should be only one fed dataset class in the file
                fed_dataset_file = Path(fed_dataset_name).expanduser().resolve()
                fed_dataset_name = None
            else:
                # of the form /path/to/dataset_file_stem.class_name
                fed_dataset_file, fed_dataset_name = fed_dataset_name.rsplit(".", 1)
                fed_dataset_file = Path(fed_dataset_file + ".py").expanduser().resolve()
            assert fed_dataset_file.exists(), (
                f"Dataset file {fed_dataset_file} not found. "
                "Please check if the dataset file exists and is a .py file, "
                "or of the form ``/path/to/dataset_file_stem.class_name``"
            )
            module = load_module_from_file(fed_dataset_file)
            # the custom federated dataset should be added to the dataset pool
            # using the decorator @register_fed_dataset
            new_fed_dataset_names = [item for item in list_fed_dataset() if item not in builtin_fed_dataset_names]
            if fed_dataset_name is None:
                # only one fed dataset class in `new_fed_dataset_names`
                # after `load_module_from_file`
                if len(new_fed_dataset_names) == 0:
                    raise ValueError(
                        f"No federated dataset found in {fed_dataset_file}. "
                        "Please check if the federated dataset is registered using "
                        "the decorator ``@register_fed_dataset`` from ``fl_sim.data_processing``"
                    )
                elif len(new_fed_dataset_names) > 1:
                    raise ValueError(
                        f"Multiple federated datasets found in {fed_dataset_file}. "
                        "Please split the federated datasets into different files, "
                        "or pass the federated dataset name in the form "
                        "``/path/to/dataset_file_stem.class_name``"
                    )
                fed_dataset_name = new_fed_dataset_names[0]
            else:
                if fed_dataset_name not in new_fed_dataset_names:
                    raise ValueError(
                        f"Federated dataset {fed_dataset_name} not found in {fed_dataset_file}. "
                        "Please check if the federated dataset is registered using "
                        "the decorator ``@register_fed_dataset`` from ``fl_sim.data_processing``"
                    )
            fed_dataset_cls = get_fed_dataset(fed_dataset_name)
        init_params = dict()
        if "datadir" in args:
            init_params["datadir"] = args.pop("datadir")
        if "seed" in args:
            init_params["seed"] = args.pop("seed")
        if "transform" in args:
            init_params["transform"] = args.pop("transform")

        ds = fed_dataset_cls(**init_params, **args)

        if len(args) > 0:
            init_params["kwargs"] = args

        init_params["name"] = fed_dataset_name
        obj = cls(**init_params)  # not used

        return ds
