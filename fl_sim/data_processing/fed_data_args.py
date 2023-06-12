"""
"""

import importlib
import inspect
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

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
        name of built-in dataset, or path to the custom dataset file (.py file)
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

        """
        return self._create_fed_dataset_from_args(self.to_dict())

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
                "or path to the custom dataset file (.py file)"
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
        if fed_dataset_name.endswith(".py"):
            assert Path(
                fed_dataset_name
            ).exists(), f"dataset file {fed_dataset_name} does not exist"
            assert Path(
                fed_dataset_name
            ).is_absolute(), f"dataset file {fed_dataset_name} is not absolute"
            spec = importlib.util.spec_from_file_location(
                Path(fed_dataset_name).stem, fed_dataset_name
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
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
        else:
            fed_dataset_cls = importlib.import_module(
                "fl_sim.data_processing"
            ).__dict__[fed_dataset_name]
        print(fed_dataset_cls)
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
