"""
"""

from typing import Iterable, Union, Optional

import torch
from deprecated import deprecated
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer
from torch_ecg.utils import add_docstring

from .base import AL_SGD_VR, AL_SGD


__all__ = [
    "FedPD_VR",
    "FedPD_SGD",
]


class FedPD_VR(AL_SGD_VR):
    """Local optimizer for ``FedPD`` using SGD with variance reduction.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.Parameter]
        Model parameters to be optimized.
    lr : float, default: 1e-3
        Learning rate.
    mu : float, default: 1.0
        The (penalty) coeff. of the augmented Lagrangian term.

    """

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1.0,
    ) -> None:
        super().__init__(params, lr=lr, mu=mu, momentum=0)


@add_docstring(
    FedPD_VR.__doc__.replace(
        "Local optimizer for ``FedPD`` using SGD with variance reduction.",
        "Local optimizer for ``FedPD`` using SGD.",
    )
)
class FedPD_SGD(AL_SGD):
    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1.0,
    ) -> None:
        super().__init__(params, lr=lr, mu=mu, momentum=0)


@deprecated(reason="This class is deprecated, use FedPD_VR instead.")
class _FedPD_VR(Optimizer):
    """Local optimizer for ``FedPD`` using SGD with variance reduction.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.Parameter]
        The parameters to optimize or dicts defining parameter groups.
    lr : float, default: 1e-3
        Learning rate.
    mu : float, default: 1.0
        The (penalty) coeff. of the augmented Lagrangian term.
    freq_1 : int, default: 10
        The frequency of the inner loop.
    freq_2 : int, default: 10
        The frequency of the outer loop.

    """

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1.0,
        freq_1: int = 10,
        freq_2: int = 10,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, freq_1=freq_1 + 1, mu=mu, freq_2=freq_2)
        self.counter_in = 0
        self.counter_out = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.

        Returns
        -------
        loss : torch.Tensor, optional
            The loss value evaluated by the closure.

        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq_1 = group["freq_1"]
            freq_2 = group["freq_2"]
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    param_state["x_0"] = torch.zeros_like(p.data)
                    param_state["g_ex"] = torch.zeros_like(p.data)
                    param_state["lambda"] = torch.zeros_like(p.data)

                x_0 = param_state["x_0"]
                g_ex = param_state["g_ex"]
                lamb = param_state["lambda"]

                if self.counter_in == 0:
                    if not (self.flag):  # first iteration, initialize
                        x_0.copy_(p.data)  # x_0 = x_i
                    else:  # after the first iteration
                        temp = p.data.clone().detach()
                        p.data.copy_(x_0)
                        x_0.copy_(temp)

                    if self.counter_out == 0:
                        g_ex.fill_(0)  # g_ex = 0

                g_ex.add_(d_p)  # g_ex = g_ex + (h-h')

                if self.counter_in > 0:  # first inner loop, only switch x_0 and x_i
                    p.data.add_(p.data - x_0, alpha=-group["lr"] * mu)
                    p.data.add_(g_ex + lamb, alpha=-group["lr"])

                if (
                    self.counter_in + 1 == freq_1
                ):  # last inner loop, perform update on lambda and x_0
                    lamb.add_(p.data - x_0, alpha=mu)
                    x_0.copy_(p.data)
                    p.data.add_(lamb, alpha=1.0 / mu)

        self.flag = True
        self.counter_in += 1
        if self.counter_in == freq_1:
            self.counter_in = 0
            self.counter_out += 1
            if self.counter_out == freq_2:
                self.counter_out = 0

        return loss


@deprecated(reason="This class is deprecated, use FedPD_SGD instead")
class _FedPD_SGD(Optimizer):
    """Local optimizer for ``FedPD`` using SGD.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.Parameter]
        The parameters to optimize or dicts defining parameter groups.
    lr : float, default: 1e-3
        Learning rate.
    mu : float, default: 1.0
        The (penalty) coeff. of the augmented Lagrangian term.
    freq : int, default: 10
        The frequency of the inner loop.

    """

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1.0,
        freq: int = 10,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, freq=freq + 1, mu=mu)
        self.counter_in = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.

        Returns
        -------
        loss : torch.Tensor, optional
            The loss value evaluated by the closure.

        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq = group["freq"]
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    # print("inner_init")
                    param_state["x_0"] = torch.zeros_like(p.data)
                    param_state["lambda"] = torch.zeros_like(p.data)

                x_0 = param_state["x_0"]
                lamb = param_state["lambda"]

                if self.counter_in == 0:
                    if not (self.flag):  # first iteration, initialize
                        x_0.copy_(p.data)  # x_0 = x_i
                    else:  # after the first iteration
                        temp = p.data.clone().detach()
                        p.data.copy_(x_0)
                        x_0.copy_(temp)

                if self.counter_in > 0:  # first inner loop, only switch x_0 and x_i
                    p.data.add_(p.data - x_0, alpha=-group["lr"] * mu)
                    p.data.add_(d_p + lamb, alpha=-group["lr"])

                if (
                    self.counter_in + 1 == freq
                ):  # last inner loop, perform update on lambda and x_0
                    lamb.add_(p.data - x_0, alpha=mu)
                    x_0.copy_(p.data)
                    p.data.add_(lamb, alpha=1.0 / mu)

        self.flag = True
        self.counter_in += 1
        if self.counter_in == freq:
            self.counter_in = 0

        return loss


# -----------------------------
# the following for comparison
# copied from the original repo


@deprecated
class PSVRG(Optimizer):
    """
    FedProx with variance reduction
    """

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1.0,
        freq: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        params: iterable of dict or Parameter,
            the parameters to optimize
        lr: float, default: 1e-3,
            the learning rate
        mu: float, default: 1.0,
            the hyperparameter for the update
        freq: int, default: 10,
            the frequency of the inner loop

        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, freq=freq, mu=mu)
        self.counter = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> Optional[Tensor]:
        """
        Performs a single optimization step.

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
            freq = group["freq"]
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    param_state["x_0"] = torch.zeros_like(p.data)
                    param_state["g_ex"] = torch.zeros_like(p.data)

                x_0 = param_state["x_0"]
                g_ex = param_state["g_ex"]

                if self.counter == 0:
                    x_0.copy_(p.data)
                    g_ex.fill_(0)

                g_ex.add_(d_p)

                p.data.add_(p.data - x_0, alpha=-group["lr"] * mu)
                p.data.add_(g_ex, alpha=-group["lr"])

        self.flag = True
        self.counter += 1
        if self.counter == freq:
            self.counter = 0
        return loss


@deprecated
class PSGD(Optimizer):
    """
    FedProx
    """

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1.0,
        freq: int = 2,
    ) -> None:
        """

        Parameters
        ----------
        params: iterable of dict or Parameter,
            the parameters to optimize
        lr: float, default: 1e-3,
            the learning rate
        mu: float, default: 1.0,
            the hyperparameter for the update
        freq: int, default: 2,
            the frequency of the inner loop
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, mu=mu, freq=freq)
        self.counter = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> Optional[Tensor]:
        """Performs a single optimization step.

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
            mu = group["mu"]
            freq = group["freq"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    param_state["x_0"] = torch.zeros_like(p.data)

                x_0 = param_state["x_0"]

                if self.counter == 0:
                    x_0.copy_(p.data)

                p.data.add_(p.data - x_0, alpha=-group["lr"] * mu)
                p.data.add_(d_p, alpha=-group["lr"])

        self.flag = True
        self.counter += 1
        if self.counter == freq:
            self.counter = 0
        return loss
