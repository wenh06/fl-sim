"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

import numpy as np
import pytest

from fl_sim.models import MLP
from fl_sim.regularizers import L1Norm, L2Norm, L2NormSquared, LInfNorm, NullRegularizer, get_regularizer


def test_regularizers():
    """ """
    model = MLP(10, 10, 10, ndim=1)
    model_params = list(model.parameters())

    regularizer = get_regularizer("l1")
    assert isinstance(regularizer, L1Norm)
    assert np.allclose(regularizer.eval(model_params), regularizer.eval(model.parameters()))
    assert len(regularizer.prox_eval(model_params)) == len(model_params)

    regularizer = get_regularizer("l2")
    assert isinstance(regularizer, L2Norm)
    assert np.allclose(regularizer.eval(model_params), regularizer.eval(model.parameters()))
    assert len(regularizer.prox_eval(model_params)) == len(model_params)

    regularizer = get_regularizer("l2squared")
    assert isinstance(regularizer, L2NormSquared)
    assert np.allclose(regularizer.eval(model_params), regularizer.eval(model.parameters()))
    assert len(regularizer.prox_eval(model_params)) == len(model_params)

    regularizer = get_regularizer("null")
    assert isinstance(regularizer, NullRegularizer)
    assert np.allclose(regularizer.eval(model_params), regularizer.eval(model.parameters()))
    assert len(regularizer.prox_eval(model_params)) == len(model_params)

    regularizer = get_regularizer("linf")
    assert isinstance(regularizer, LInfNorm)
    assert np.allclose(regularizer.eval(model_params), regularizer.eval(model.parameters()))
    with pytest.raises(NotImplementedError):
        regularizer.prox_eval(model_params)
