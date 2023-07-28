from typing import Union, Iterable

from torch.nn.parameter import Parameter

from fl_sim.optimizers.base import ProxSGD
from fl_sim.optimizers._register import register_optimizer


__all__ = [
    "CustomOptimizer",
]


@register_optimizer()
class CustomOptimizer(ProxSGD):
    """Local optimizer for ``CustomOptimizer`` using ``ProxSGD``.

    Copied from ``FedProxOptimizer``.

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
    1. https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py
    2. https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pggd.py

    The ``gold`` (reference 2) is not re-implemented yet.

    """

    __name__ = "CustomOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1e-2,
    ) -> None:
        self.mu = mu
        super().__init__(params, lr=lr, prox=mu, momentum=0)
