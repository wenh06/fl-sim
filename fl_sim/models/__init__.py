"""
fl_sim.models
==================

This module contains built-in simple models.

.. contents:: fl_sim.models
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

"""

from .nn import (
    MLP,
    FedPDMLP,
    CNNMnist,
    CNNFEMnist,
    CNNFEMnist_Tiny,
    CNNCifar,
    CNNCifar_Small,
    CNNCifar_Tiny,
    RNN_OriginalFedAvg,
    RNN_StackOverFlow,
    RNN_Sent140,
    RNN_Sent140_LITE,
    ResNet18,
    ResNet10,
    LogisticRegression,
    SVC,
    SVR,
)
from .utils import reset_parameters, top_n_accuracy


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
    "ResNet18",
    "ResNet10",
    "LogisticRegression",
    "SVC",
    "SVR",
    "reset_parameters",
    "top_n_accuracy",
]
