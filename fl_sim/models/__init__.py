"""
built-in simple models
"""

from .nn import (
    MLP,
    FedPDMLP,
    CNNMnist,
    CNNFEMnist,
    CNNFEMnist_Tiny,
    CNNCifar,
    RNN_OriginalFedAvg,
    RNN_StackOverFlow,
    RNN_Sent140,
    ResNet18,
    ResNet10,
    LogisticRegression,
    SVC,
    SVR,
)


__all__ = [
    "MLP",
    "FedPDMLP",
    "CNNMnist",
    "CNNFEMnist",
    "CNNFEMnist_Tiny",
    "CNNCifar",
    "RNN_OriginalFedAvg",
    "RNN_StackOverFlow",
    "RNN_Sent140",
    "ResNet18",
    "ResNet10",
    "LogisticRegression",
    "SVC",
    "SVR",
]
