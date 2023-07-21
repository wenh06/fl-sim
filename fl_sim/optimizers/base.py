"""
"""

import warnings
from typing import Iterable, Union, Optional, Any

import torch  # noqa: F401
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer
from torch_ecg.utils import add_docstring, remove_parameters_returns_from_docstring

from . import functional as F
from ._register import register_optimizer


__all__ = [
    "ProxSGD_VR",
    "ProxSGD",
    "SGD_VR",
    "AL_SGD_VR",
    "AL_SGD",
]


_prox_sgd_params_doc = """
    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
        The model parameters to optimize.
    lr : float, default 1e-3
        Learning rate.
    momentum : float, default 1e-3
        Momentum factor.
    dampening : float, default 0
        dampening for momentum
    weight_decay : float, default 0
        Weight decay factor (L2 penalty).
    nesterov : bool, default False
        If True, enables Nesterov momentum.
    prox : float, default 0.1
        The (penalty) coeff. of the proximal term,
        i.e. the term :math:`\\rho` in

        .. math::

            \\argmin_x \\{ f(x) + \\dfrac{\\rho}{2} \\lVert x-v \\rVert_2^2 \\}

    """


@register_optimizer()
@add_docstring(_prox_sgd_params_doc, mode="append")
class ProxSGD_VR(Optimizer):
    """Proximal Stochastic Gradient Descent with Variance Reduction.

    Using SGD to solve the proximal problem:

        .. math::
            \\DeclareMathOperator*{\\argmin}{arg\\,min}
            \\operatorname{prox}_{\\rho f}(v) =
            \\argmin_x \\{f(x) + \\dfrac{\\rho}{2} \\lVert x-v \\rVert_2^2\\}

    when it does not have a closed form solution.

    """

    __name__ = "ProxSGD_VR"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        momentum: float = 1e-3,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        prox: float = 0.1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if prox < 0.0:
            raise ValueError(f"Invalid prox value: {prox}")
        if prox * lr >= 1:
            warnings.warn(
                f"prox * lr = {prox * lr:.3f} >= 1 with prox = {prox}, lr = {lr}, "
                f"you may encounter gradient exploding.",
                RuntimeWarning,
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            prox=prox,
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
            i.e. the term :math:`v` in

            .. math::

                \\argmin_x \\{ f(x) + \\dfrac{\\rho}{2} \\lVert x-v \\rVert_2^2 \\}

        variance_buffer : Iterable[torch.Tensor], optional
            The variance buffer of the local weights,
            for the variance-reduced algorithms.
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss : torch.Tensor, optional
            Loss value, returned only if `closure` is not ``None``.

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
            prox = group["prox"]
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

            F.prox_vr_sgd(
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
                prox=prox,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


@register_optimizer()
@add_docstring(_prox_sgd_params_doc, mode="append")
class ProxSGD(ProxSGD_VR):
    """Proximal Stochastic Gradient Descent.

    Using SGD to solve the proximal problem:

    .. math::

        \\DeclareMathOperator*{\\argmin}{arg\\,min}
        \\operatorname{prox}_{\\rho f}(v) =
        \\argmin_x \\{ f(x) + \\dfrac{\\rho}{2} \\lVert x-v \\rVert_2^2 \\}

    when it does not have a closed form solution.

    """

    __name__ = "ProxSGD"

    @torch.no_grad()
    @add_docstring(
        remove_parameters_returns_from_docstring(
            ProxSGD_VR.step.__doc__, parameters="variance_buffer"
        )
    )
    def step(
        self,
        local_weights: Iterable[Parameter],
        closure: Optional[callable] = None,
        **kwargs: Any,
    ) -> Optional[Tensor]:
        return super().step(local_weights, None, closure)


@register_optimizer()
class SGD_VR(ProxSGD_VR):
    """Stochastic Gradient Descent with Variance Reduction.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
        The parameters to optimize or dicts defining parameter groups.
    lr : float, default 1e-3
        Learning rate.
    momentum : float, default 1e-3
        Momentum factor.
    dampening : float, default 0
        Dampening for momentum.
    weight_decay : float, default 0
        Weight decay factor (L2 penalty).
    nesterov: bool, default False
        If True, enables Nesterov momentum

    """

    __name__ = "SGD_VR"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        momentum: float = 1e-3,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov, 0.0)
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.defaults = defaults

    @torch.no_grad()
    @add_docstring(
        remove_parameters_returns_from_docstring(
            ProxSGD_VR.step.__doc__, parameters="local_weights"
        )
    )
    def step(
        self,
        variance_buffer: Iterable[Tensor],
        closure: Optional[callable] = None,
        **kwargs: Any,
    ) -> Optional[Tensor]:
        return super().step(None, variance_buffer, closure)


_al_sgd_vr_params_doc = """
    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
        The parameters to optimize or dicts defining parameter groups.
    lr : float, default 1e-3
        Learning rate.
    momentum : float, default 1e-3
        Momentum factor.
    dampening : float, default 0
        Dampening for momentum.
    weight_decay : float, default 0
        Weight decay factor (L2 penalty).
    nesterov: bool, default False
        If True, enables Nesterov momentum
    mu : float, default 0.1
        The (penalty) coeff. of the augmented Lagrangian term,
        i.e. the term :math:`\\mu` in

        .. math::

            \\mathcal{L}_{\\mu}(x, x_0, \\lambda)

    """


@register_optimizer()
@add_docstring(_al_sgd_vr_params_doc, mode="append")
class AL_SGD_VR(Optimizer):
    """Augmented Lagrangian Stochastic Gradient Descent with Variance Reduction.

    Using SGD to solve the augmented Lagrangian problem:

    .. math::

        \\DeclareMathOperator*{\\argmin}{arg\\,min}
        \\argmin_x \\mathcal{L}_{\\mu}(x, x_0, \\lambda) =
        \\argmin_x \\{ f(x) + \\langle \\lambda, x-x_0 \\rangle + \\dfrac{1}{2\\mu} \\lVert x-x_0 \\rVert_2^2 \\}

    when it does not have a closed form solution.

    """

    __name__ = "AL_SGD_VR"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: float = False,
        mu: float = 1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if mu < 0.0:
            raise ValueError(f"Invalid mu value: {mu}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            mu=mu,
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
        dual_weights: Iterable[Parameter],
        variance_buffer: Iterable[Tensor],
        closure: Optional[callable] = None,
        **kwargs: Any,
    ) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        local_weights : Iterable[torch.nn.parameter.Parameter]
            The (init) local weights, i.e. the term :math:`x_0` in

            .. math::

                \\mathcal{L}_{\\mu}(x, x_0, \\lambda)

        dual_weights : Iterable[torch.nn.parameter.Parameter]
            The weights of dual variables,
            i.e. the term :math:`\\lambda` in

            .. math::

                \\mathcal{L}_{\\mu}(x, x_0, \\lambda)

        variance_buffer : Iterable[torch.Tensor]
            The variance buffer of the local weights,
            for the variance-reduced algorithms.

        Returns
        -------
        loss : torch.Tensor, optional
            Loss value, returned only if `closure` is not ``None``.

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
            mu = group["mu"]
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

            F.al_sgd(
                params_with_grad,
                local_weights,
                dual_weights,
                d_p_list,
                momentum_buffer_list,
                weight_decay,
                momentum,
                lr,
                dampening,
                nesterov,
                mu,
            )

            # update momentum_buffers, gradient_variance_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


@register_optimizer()
@add_docstring(_al_sgd_vr_params_doc, mode="append")
class AL_SGD(AL_SGD_VR):
    """Augmented Lagrangian Stochastic Gradient Descent.

    Using SGD to solve the augmented Lagrangian problem:

    .. math::

        \\DeclareMathOperator*{\\argmin}{arg\\,min}
        \\argmin_x \\mathcal{L}_{\\mu}(x, x_0, \\lambda) =
        \\argmin_x \\{ f(x) + \\langle \\lambda, x-x_0 \\rangle + \\dfrac{1}{2\\mu} \\lVert x-x_0 \\rVert_2^2 \\}

    when it does not have a closed form solution.

    """

    @torch.no_grad()
    @add_docstring(
        remove_parameters_returns_from_docstring(
            AL_SGD_VR.step.__doc__, parameters="variance_buffer"
        )
    )
    def step(
        self,
        local_weights: Iterable[Parameter],
        dual_weights: Iterable[Parameter],
        closure: Optional[callable] = None,
        **kwargs: Any,
    ) -> Optional[Tensor]:
        return super().step(local_weights, dual_weights, None, closure)
