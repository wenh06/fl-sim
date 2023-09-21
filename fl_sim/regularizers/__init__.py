"""
fl_sim.regularizers
===================

This module contains the regularizers for the optimization problems.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: fl_sim.regularizers

.. autosummary::
    :toctree: generated/
    :recursive:

    get_regularizer
    Regularizer
    L1Norm
    L2Norm
    L2NormSquared
    LInfNorm
    NullRegularizer

"""

from .regularizers import L1Norm, L2Norm, L2NormSquared, LInfNorm, NullRegularizer, Regularizer, get_regularizer

__all__ = [
    "get_regularizer",
    "Regularizer",
    "L1Norm",
    "L2Norm",
    "L2NormSquared",
    "LInfNorm",
    "NullRegularizer",
]
