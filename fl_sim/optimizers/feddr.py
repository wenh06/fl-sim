"""
"""

from typing import Iterable, Union

from torch.nn.parameter import Parameter

from ._register import register_optimizer
from .base import ProxSGD

__all__ = [
    "FedDROptimizer",
]


@register_optimizer()
class FedDROptimizer(ProxSGD):
    """Local optimizer for ``FedDR`` algorithm.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
        The parameters to optimize or dicts defining parameter groups.
    lr : float, default 0.01
        Learning rate.
    eta : float, default 0.1
        Reciprocal coeff. of the proximal term.

    """

    __name__ = "FedDROptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        eta: float = 1.0,
    ) -> None:
        self.eta = eta
        super().__init__(params, lr=lr, prox=1 / eta, momentum=0)
