"""
"""

import importlib
import inspect
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

from .fed_dataset import FedDataset
from ._register import list_fed_dataset, get_fed_dataset


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
        fed_dataset_name = args["name"]
        # if args["name"] is a path to a dataset file, import the dataset from the file
        if fed_dataset_name in list_fed_dataset():
            fed_dataset_cls = get_fed_dataset(fed_dataset_name)
        else:
            if fed_dataset_name.endswith(".py"):
                # is a .py file
                fed_dataset_file = fed_dataset_name
                fed_dataset_name = None
            else:
                # of the form /path/to/dataset.class_name
                fed_dataset_file, fed_dataset_name = fed_dataset_name.rsplit(".", 1)
                fed_dataset_file += ".py"
            fed_dataset_file = Path(fed_dataset_file).resolve()
            assert (
                fed_dataset_file.exists()
            ), f"dataset file {fed_dataset_file} does not exist"
            spec = importlib.util.spec_from_file_location(
                fed_dataset_file.stem, fed_dataset_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if fed_dataset_name is None:
                for attr in dir(module):
                    candidate = getattr(module, attr)
                    if not inspect.isclass(candidate) or not issubclass(
                        candidate, FedDataset
                    ):
                        continue
                    if candidate.__module__.startswith("fl_sim"):
                        continue
                    fed_dataset_name = candidate.__name__
                    fed_dataset_cls = candidate
                    break
                assert fed_dataset_name is not None, (
                    f"dataset file {fed_dataset_file} does not contain a "
                    "subclass of FedDataset"
                )
            else:
                if hasattr(module, fed_dataset_name):
                    fed_dataset_cls = getattr(module, fed_dataset_name)
                else:
                    raise ValueError(
                        f"dataset file {fed_dataset_file} does not contain "
                        f"a class named {fed_dataset_name}"
                    )
        obj = cls(
            name=fed_dataset_name,
            datadir=args["datadir"],
            transform=args["transform"],
            seed=args["seed"],
        )

        ds = fed_dataset_cls(
            datadir=obj.datadir, transform=obj.transform, seed=obj.seed
        )

        return ds
