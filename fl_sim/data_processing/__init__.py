"""
fl_sim.data_processing
=======================

This module contains federated datasets and data processing utilities.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: fl_sim.data_processing

Base classes
------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    FedDataset
    FedVisionDataset
    FedNLPDataset

Vision datasets
--------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    FedCIFAR
    FedCIFAR100
    FedEMNIST
    FedMNIST
    FedRotatedCIFAR10
    FedRotatedMNIST
    FedProxFEMNIST
    FedProxMNIST
    FedTinyImageNet

NLP datasets
--------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    FedShakespeare
    FedProxSent140

Synthetic datasets
--------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    FedSynthetic

LibSVM datasets
--------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    FedLibSVMDataset

Dataset registry utilities
--------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    register_fed_dataset
    list_fed_dataset
    get_fed_dataset

"""

from ._register import get_fed_dataset, list_fed_dataset, register_fed_dataset
from .fed_cifar import FedCIFAR, FedCIFAR100
from .fed_data_args import FedDataArgs
from .fed_dataset import FedDataset, FedNLPDataset, FedVisionDataset
from .fed_emnist import FedEMNIST
from .fed_mnist import FedMNIST
from .fed_rotated_cifar10 import FedRotatedCIFAR10
from .fed_rotated_mnist import FedRotatedMNIST
from .fed_shakespeare import FedShakespeare
from .fed_synthetic import FedSynthetic

# additional datasets
from .fed_tiny_imagenet import FedTinyImageNet
from .fedprox_femnist import FedProxFEMNIST
from .fedprox_mnist import FedProxMNIST
from .fedprox_sent140 import FedProxSent140

# from .leaf_sent140 import LeafSent140
from .libsvm_datasets import FedLibSVMDataset, libsvmread

builtin_datasets = list_fed_dataset()


__all__ = [
    "FedDataArgs",
    # base classes
    "FedDataset",
    "FedVisionDataset",
    "FedNLPDataset",
    # datasets from FedML
    "FedCIFAR",
    "FedCIFAR100",
    "FedEMNIST",
    "FedMNIST",
    "FedRotatedCIFAR10",
    "FedRotatedMNIST",
    "FedShakespeare",
    "FedSynthetic",
    # additional datasets
    "FedTinyImageNet",
    # datasets from FedProx
    "FedProxFEMNIST",
    "FedProxMNIST",
    "FedProxSent140",
    # libsvm datasets
    "FedLibSVMDataset",
    "libsvmread",
    "list_fed_dataset",
    "builtin_datasets",
    "get_fed_dataset",
    "register_fed_dataset",
]
