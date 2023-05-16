"""
"""

from typing import Iterable, Union, Optional, Tuple

import torch  # noqa: F401
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer
from deprecated import deprecated

from .base import ProxSGD


__all__ = [
    "pFedMeOptimizer",
]


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


@deprecated(reason="This class is deprecated, use ``pFedMeOptimizer`` instead.")
class _pFedMeOptimizer(Optimizer):
    """Legacy :class:`pFedMeOptimizer`.

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

    __name__ = "_pFedMeOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 0.01,
        lamda: float = 0.1,
        mu: float = 1e-3,
    ) -> None:
        # self.local_weights = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super().__init__(params, defaults)

    def step(
        self,
        local_weights: Iterable[Parameter],
        closure: Optional[callable] = None,
    ) -> Tuple[Iterable[Parameter], Optional[Tensor]]:
        """Performs a single optimization step.

        Parameters
        ----------
        local_weights : Iterable[torch.nn.Parameter]
            The local weights updated by the server.
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        List[torch.nn.Parameter]
            The list of tensors for the updated parameters.
        torch.Tensor, optional
            The loss value evaluated by the closure.

        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, lw in zip(group["params"], local_weights):
                p.data = p.data - group["lr"] * (
                    p.grad.data
                    + group["lamda"] * (p.data - lw.data)
                    + group["mu"] * p.data
                )
        return group["params"], loss

    def update_param(
        self,
        local_weights: Iterable[Parameter],
        closure: Optional[callable] = None,
    ) -> Iterable[Parameter]:
        """Update the parameters of the model.

        Parameters
        ----------
        local_weights : Iterable[torch.nn.Parameter]
            The local weights updated by the server.
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        List[torch.nn.Parameter]
            The list of tensors for the updated parameters.

        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, lw in zip(group["params"], local_weights):
                p.data = lw.data.clone()
        # return  p.data
        return group["params"]


# -----------------------------
# the following for comparison
# copied from the original pFedMe repo


@deprecated
class FEDLOptimizer(Optimizer):

    __name__ = "FEDLOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 0.01,
        server_grads: Optional[Tensor] = None,
        pre_grads: Optional[Tensor] = None,
        eta: float = 0.1,
    ) -> None:
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eta=eta)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> Optional[Tensor]:
        """
        Parameters
        ----------
        closure: callable, optional,
            a closure that reevaluates the model and returns the loss

        Returns
        -------
        loss: Tensor, optional,
            the loss after the step

        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            i = 0
            for p in group["params"]:
                p.data = p.data - group["lr"] * (
                    p.grad.data
                    + group["eta"] * self.server_grads[i]
                    - self.pre_grads[i]
                )
                # p.data.add_(p.grad.data, alpha=-group["lr"])
                i += 1
        return loss


@deprecated
class APFLOptimizer(Optimizer):
    __name__ = "APFLOptimizer"

    def __init__(
        self, params: Iterable[Union[dict, Parameter]], lr: float = 0.01
    ) -> None:
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(
        self, closure: Optional[callable] = None, beta: float = 1.0, n_k: float = 1.0
    ) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # print(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(d_p, alpha=-group["lr"])
        return loss
