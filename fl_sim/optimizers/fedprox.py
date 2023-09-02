"""
"""

from typing import Union, Iterable

from torch.nn.parameter import Parameter
from torch_ecg.utils import add_docstring

from .base import ProxSGD, ProxSGD_VR
from ._register import register_optimizer


__all__ = [
    "FedProxOptimizer",
    "FedProx_VR",
]


@register_optimizer()
class FedProxOptimizer(ProxSGD):
    """Local optimizer for ``FedProx`` using ``ProxSGD``.

    The original implementation is in [1]_, [2]_.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
        The parameters to optimize or dicts defining parameter groups.
    lr : float, default: 1e-3
        Learning rate.
    mu : float, default 0.1
        Coeff. of the proximal term.

    References
    ----------
    .. [1] https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py
    .. [2] https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pggd.py

    .. note::

        The ``gold`` [2]_ is not re-implemented yet.

    """

    __name__ = "FedProxOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1e-2,
    ) -> None:
        self.mu = mu
        super().__init__(params, lr=lr, prox=mu, momentum=0)


@register_optimizer()
@add_docstring(
    FedProxOptimizer.__doc__.replace(
        "Local optimizer for ``FedProx`` using ``ProxSGD``.",
        "Local optimizer for ``FedProx`` using ``ProxSGD`` with variance reduction.",
    )
)
class FedProx_VR(ProxSGD_VR):

    __name__ = "FedProx_VR"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1e-2,
    ) -> None:
        self.mu = mu
        super().__init__(params, lr=lr, prox=mu, momentum=0)
