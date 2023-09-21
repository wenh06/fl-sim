"""
"""

from typing import Any, Iterable, Optional, Union

import torch  # noqa: F401
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer

from . import functional as F
from ._register import register_optimizer

__all__ = [
    "pFedMacOptimizer",
]


@register_optimizer()
class pFedMacOptimizer(Optimizer):
    """Local optimizer for ``pFedMac`` via maximizing correlation (Mac)
    using ``SGD`` (with variance reduction).

    Mathematical definition:

    .. math::

        \\DeclareMathOperator*{\\argmin}{arg\\,min}
        \\argmin_x \\{ f(x) - \\lambda \\langle x, x_0 \\rangle \\}

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
        The parameters to optimize or dicts defining parameter groups.
    lr : float, default: 1e-3
        Learning rate.
    momentum : float, default 1e-3
        Momentum factor.
    dampening : float, default 0
        Dampening for momentum.
    weight_decay : float, default 0
        Weight decay factor (L2 penalty).
    nesterov : bool, default False
        If True, enables Nesterov momentum.
    lam : float, default 0.1
        The (penalty) coeff. of the maximizing correlation term,
        i.e. the term :math:`\\lambda` in

        .. math::

            \\lambda \\langle x, x_0 \\rangle

    """

    __name__ = "pFedMacOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        momentum: float = 1e-3,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        lam: float = 0.1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if lam < 0.0:
            raise ValueError(f"Invalid lam value: {lam}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            lam=lam,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(
        self,
        local_weights: Iterable[Parameter],
        variance_buffer: Optional[Iterable[Tensor]] = None,
        closure: Optional[callable] = None,
        **kwargs: Any,
    ) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        local_weights : Iterable[torch.nn.parameter.Parameter]
            The local weights updated by the local optimizer,
            or of the previous iteration,
            i.e. the term :math:`x_0` in

            .. math::

                \\lambda \\langle x, x_0 \\rangle

        variance_buffer : Iterable[torch.Tensor], optional
            The variance buffer of the local weights,
            for the variance-reduced algorithms.
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss : torch.Tensor, optional
            The loss value evaluated by the closure.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lam = group["lam"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            F.mac_sgd(
                params_with_grad,
                local_weights,
                variance_buffer,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                lam=lam,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss
