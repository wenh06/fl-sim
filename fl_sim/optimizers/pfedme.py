"""
For original implementation, see:
https://github.com/CharlieDinh/pFedMe
"""

from typing import Iterable, Union

from torch.nn import Parameter

from .base import ProxSGD
from ._register import register_optimizer


__all__ = [
    "pFedMeOptimizer",
]


@register_optimizer()
class pFedMeOptimizer(ProxSGD):
    """Local optimizer for ``pFedMe`` using :class:`ProxSGD`.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.Parameter]
        Model parameters to be optimized.
    lr : float, default 0.01
        Learning rate.
    lamda : float, default 0.1
        Coeff. of the proximal term.
    mu : float, default 1e-3
        Momentum coeff.

    """

    __name__ = "pFedMeOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 0.01,
        lamda: float = 0.1,
        mu: float = 1e-3,
    ) -> None:
        self.lamda = lamda
        self.mu = mu
        super().__init__(params, lr=lr, prox=lamda, momentum=mu, nesterov=True)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", True)
