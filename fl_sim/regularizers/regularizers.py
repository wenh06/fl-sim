"""
this file is forked from `FedDR <https://github.com/unc-optimization/FedDR/tree/main/FedDR/flearn/regularizers>`_.
"""

import re
from abc import ABC, abstractmethod
from math import sqrt
from typing import Iterable, List, Optional

from torch.nn.parameter import Parameter
from torch_ecg.utils import ReprMixin, add_docstring


__all__ = [
    "get_regularizer",
    "Regularizer",
    "L1Norm",
    "L2Norm",
    "L2NormSquared",
    "LInfNorm",
    "NullRegularizer",
]


class Regularizer(ReprMixin, ABC):
    """Regularizer base class.

    Parameters
    ----------
    coeff : float, default 1.0
        The coefficient of the regularizer.

    """

    __name__ = "Regularizer"

    def __init__(self, coeff: float = 1.0) -> None:
        self.coeff = coeff

    @abstractmethod
    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        """Evaluate the regularizer on the given parameters.

        Parameters
        ----------
        params : Iterable[torch.nn.parameter.Parameter]
            The parameters to be evaluated on.
        coeff : float, optional
            The coefficient of the regularizer.
            If None, use the default value.

        """
        raise NotImplementedError

    @abstractmethod
    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        """Evaluate the proximity operator of the regularizer
        on the given parameters.

        i.e. evaluate the following function:

        .. math::

            \\mathrm{prox}_{\\lambda R}(\\mathbf{w})
            = \\arg\\min_{\\mathbf{u}} \\frac{1}{2s} \\lVert \\mathbf{u} - \\mathbf{w} \\rVert_2^2
            + \\lambda R(\\mathbf{u})

        where :math:`R` is the regularizer.

        Parameters
        ----------
        params : Iterable[torch.nn.parameter.Parameter]
            The parameters to be evaluated on.
        coeff : float, optional
            The coefficient of the regularizer.
            If None, use the default value.

        Returns
        -------
        Iterable[torch.nn.parameter.Parameter]
            The proximity operator of the regularizer
            evaluated on the given parameters.

        """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "coeff",
        ]


def get_regularizer(reg_type: str, reg_coeff: float = 1.0) -> Regularizer:
    """Get the regularizer by name.

    Parameters
    ----------
    reg_type : str
        The name of the regularizer.
    reg_coeff : float, default 1.0
        The coefficient of the regularizer.

    Returns
    -------
    Regularizer
        The regularizer instance.

    """
    reg_type = re.sub("regularizer|norm|[\\s\\_\\-]+", "", reg_type.lower())
    if reg_type in [
        "l1",
    ]:
        return L1Norm(reg_coeff)
    elif reg_type in [
        "l2",
    ]:
        return L2Norm(reg_coeff)
    elif reg_type in [
        "l2squared",
    ]:
        return L2NormSquared(reg_coeff)
    elif reg_type in [
        "no",
        "empty",
        "zero",
        "none",
        "null",
    ]:
        return NullRegularizer(reg_coeff)
    elif reg_type in [
        "linf",
        "inf",
        "linfinity",
        "infinity",
        "linfty",
        "infty",
    ]:
        return LInfNorm(reg_coeff)
    else:
        raise ValueError(f"Unknown regularizer type: {reg_type}")


@add_docstring(
    Regularizer.__doc__.replace(
        "Regularizer base class.",
        "Null regularizer, or equivalently the zero function.",
    )
)
class NullRegularizer(Regularizer):

    __name__ = "NullRegularizer"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        return 0.0

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        return list(params)


@add_docstring(
    Regularizer.__doc__.replace(
        "Regularizer base class.",
        "L1 norm regularizer.",
    )
)
class L1Norm(Regularizer):

    __name__ = "L1Norm"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        if coeff is None:
            coeff = self.coeff
        return coeff * sum([p.data.abs().sum().item() for p in params])

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        if coeff is None:
            coeff = self.coeff
        ret_params = [
            p.data.sign() * (p.data.abs() - coeff).clamp(min=0) for p in params
        ]
        return ret_params


@add_docstring(
    Regularizer.__doc__.replace(
        "Regularizer base class.",
        "L2 norm regularizer.",
    )
)
class L2Norm(Regularizer):

    __name__ = "L2Norm"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        if coeff is None:
            coeff = self.coeff
        return coeff * sqrt(sum([p.data.pow(2).sum().item() for p in params]))

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        if coeff is None:
            coeff = self.coeff
        _params = list(params)  # to avoid the case that params is a generator
        norm = self.eval(_params, coeff=coeff)
        coeff = max(0, 1 - coeff / norm)
        ret_params = [coeff * p.data for p in _params]
        del _params
        return ret_params


@add_docstring(
    Regularizer.__doc__.replace(
        "Regularizer base class.",
        "L2 norm squared regularizer.",
    )
)
class L2NormSquared(Regularizer):

    __name__ = "L2NormSquared"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        if coeff is None:
            coeff = self.coeff
        return coeff * sum([p.data.pow(2).sum().item() for p in params])

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        if coeff is None:
            coeff = self.coeff
        coeff = 1 / (1 + 2 * coeff)
        _params = list(params)  # to avoid the case that params is a generator
        ret_params = [coeff * p.data for p in _params]
        del _params
        return ret_params


@add_docstring(
    Regularizer.__doc__.replace(
        "Regularizer base class.",
        "L-infinity norm regularizer.",
    )
)
class LInfNorm(Regularizer):

    __name__ = "LInfNorm"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        if coeff is None:
            coeff = self.coeff
        return coeff * max([p.data.abs().max().item() for p in params])

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        if coeff is None:
            coeff = self.coeff
        _params = list(params)  # to avoid the case that params is a generator
        raise NotImplementedError("L-infinity norm is not implemented yet")
