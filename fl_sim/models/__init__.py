"""
fl_sim.models
==================

This module contains built-in simple models.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: fl_sim.models

Convolutional neural networks (CNN)
------------------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    CNNMnist
    CNNFEMnist
    CNNFEMnist_Tiny
    CNNCifar
    CNNCifar_Small
    CNNCifar_Tiny
    ResNet18
    ResNet10
    ShrinkedResNet

Recurrent neural networks (RNN)
------------------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    RNN_OriginalFedAvg
    RNN_StackOverFlow
    RNN_Sent140
    RNN_Sent140_LITE

Multilayer perceptron (MLP)
------------------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    MLP
    FedPDMLP

Linear models
------------------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    LogisticRegression
    SVC
    SVR

Utilities
------------------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    reset_parameters
    top_n_accuracy
    CLFMixin
    REGMixin
    DiffMixin

"""

from .nn import (
    MLP,
    SVC,
    SVR,
    CNNCifar,
    CNNCifar_Small,
    CNNCifar_Tiny,
    CNNFEMnist,
    CNNFEMnist_Tiny,
    CNNMnist,
    FedPDMLP,
    LogisticRegression,
    ResNet10,
    ResNet18,
    RNN_OriginalFedAvg,
    RNN_Sent140,
    RNN_Sent140_LITE,
    RNN_StackOverFlow,
    ShrinkedResNet,
)
from .utils import CLFMixin, DiffMixin, REGMixin, reset_parameters, top_n_accuracy

__all__ = [
    "MLP",
    "FedPDMLP",
    "CNNMnist",
    "CNNFEMnist",
    "CNNFEMnist_Tiny",
    "CNNCifar",
    "CNNCifar_Small",
    "CNNCifar_Tiny",
    "RNN_OriginalFedAvg",
    "RNN_StackOverFlow",
    "RNN_Sent140",
    "RNN_Sent140_LITE",
    "ShrinkedResNet",
    "ResNet18",
    "ResNet10",
    "LogisticRegression",
    "SVC",
    "SVR",
    "reset_parameters",
    "top_n_accuracy",
    "CLFMixin",
    "REGMixin",
    "DiffMixin",
]
