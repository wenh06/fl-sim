"""
"""

from typing import Iterable, List, Optional

import torch
from torch import Tensor
from torch_ecg.utils import add_docstring, remove_parameters_returns_from_docstring

__all__ = [
    "prox_vr_sgd",
    "prox_sgd",
    "al_vr_sgd",
    "al_sgd",
]


def prox_vr_sgd(
    params: List[Tensor],
    local_weights: Iterable[Tensor],
    variance_buffer: Iterable[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    prox: float,
) -> None:
    """The function that executes the proximal SGD with variance reduction.

    Mathematical definition:

    .. math::

        \\DeclareMathOperator*{\\argmin}{arg\\,min}
        \\operatorname{prox}_{\\rho f}(v) =
        \\argmin_x \\{ f(x) + \\dfrac{\\rho}{2} \\lVert x-v \\rVert_2^2 \\}

    Parameters
    ----------
    params : List[torch.Tensor]
        The parameters to optimize or dicts defining parameter groups.
    local_weights : List[Tensor]
        The local weights updated by the local optimizer,
        or of the previous iteration,
        i.e. the term :math:`v` in

        .. math::

            \\argmin_x \\{ f(x) + \\dfrac{\\rho}{2} \\lVert x-v \\rVert_2^2 \\}

    variance_buffer : List[Tensor]
        The variance buffers of the parameters,
        used for variance reduction.
    d_p_list : List[Tensor]
        The list of gradients of the parameters.
    momentum_buffer_list : List[Optional[Tensor]]
        The list of momentum buffers.
    weight_decay : float
        Weight decay factor (L2 penalty).
    momentum : float
        Momentum factor.
    lr : float
        The learning rate.
    dampening : float
        Dampening for momentum.
    nesterov : bool
        If True, enables Nesterov momentum.
    prox : float
        The (penalty) coeff. of the proximal term,

        i.e. the term :math:`\\rho` in

        .. math::

            \\argmin_x \\{ f(x) + \\dfrac{\\rho}{2} \\lVert x-v \\rVert_2^2 \\}

    """
    if local_weights is None:
        local_weights = [None] * len(params)
    if variance_buffer is None:
        variance_buffer = [None] * len(params)
    for idx, (param, lw, vb) in enumerate(zip(params, local_weights, variance_buffer)):
        d_p = d_p_list[idx]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)  # L2 regularization

        if prox != 0 and lw is not None:
            d_p = d_p.add(param - lw.detach().clone(), alpha=prox)  # proximal regularization

        if momentum != 0:
            buf = momentum_buffer_list[idx]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[idx] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if vb is not None:
            d_p = d_p.sub(vb.detach().clone())  # variance reduction

        param.add_(d_p, alpha=-lr)


@add_docstring(
    remove_parameters_returns_from_docstring(prox_vr_sgd.__doc__, parameters="variance_buffer").replace(
        "The function that executes the proximal SGD with variance reduction.",
        "The function that executes the proximal SGD.",
    )
)
def prox_sgd(
    params: List[Tensor],
    local_weights: Iterable[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    prox: float,
) -> None:
    return prox_vr_sgd(
        params,
        local_weights,
        None,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        prox=prox,
    )


def al_vr_sgd(
    params: List[Tensor],
    local_weights: List[Tensor],
    dual_weights: List[Tensor],
    variance_buffer: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    mu: float,
) -> None:
    r"""
    The function that executes the augmented Lagrangian SGD with variance reduction:
        .. math::
            \DeclareMathOperator*{\argmin}{arg\,min}
            \argmin_x \mathcal{L}_{\mu}(x, x_0, \lambda) =
            \argmin_x \{f(x) + \langle \lambda, x-x_0 \rangle + \dfrac{1}{2\mu} \lVert x-x_0 \rVert_2^2\}

    Parameters
    ----------
    params: list of dict or Parameter,
        the parameters to optimize
    local_weights: iterable of Parameter,
        the (init) local weights,
        i.e. the term `x_0` in
            .. math::
                \mathcal{L}_{\mu}(x, x_0, \lambda)
    dual_weights: iterable of Parameter,
        the weights of dual variables,
        i.e. the term `\lambda` in
            .. math::
                \mathcal{L}_{\mu}(x, x_0, \lambda)
    variance_buffer: list of Parameter, optional,
        the variance buffers of the parameters,
        used for variance reduction
    d_p_list: list of Tensor,
        the list of gradients of the parameters
    momentum_buffer_list: list of Tensor or list of None,
        the list of momentum buffers,
        works only if `momentum` > 0
    gradient_variance_buffer_list: list of Tensor or list of None,
        the list of gradient variance buffers,
        works only is `vr` is True
    weight_decay: float,
        weight decay (L2 penalty)
    momentum: float,
        momentum factor
    lr: float, default 1e-3,
        the learning rate
    dampening: float,
        dampening for momentum
    nesterov: bool,
        if True, enables Nesterov momentum
    mu: float,
        the (penalty) coeff. of the augmented Lagrangian term,
        i.e. the term `\mu` in
            .. math::
                \mathcal{L}_{\mu}(x, x_0, \lambda)

    """
    if variance_buffer is None:
        variance_buffer = [None] * len(params)
    for idx, (param, lw, dw, vb) in enumerate(zip(params, local_weights, dual_weights, variance_buffer)):
        d_p = d_p_list[idx]

        d_p = d_p.add(dw.detach().clone())

        d_p = d_p.add(param - lw.detach().clone(), alpha=1 / mu)

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)  # L2 regularization

        if momentum != 0:
            buf = momentum_buffer_list[idx]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[idx] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if vb is not None:
            d_p = d_p.sub(vb.detach().clone())

        param.add_(d_p, alpha=-lr)


@add_docstring(
    remove_parameters_returns_from_docstring(al_vr_sgd.__doc__, parameters="variance_buffer").replace(
        "The function that executes the augmented Lagrangian SGD with variance reduction:",
        "The function that executes the augmented Lagrangian SGD:",
    )
)
def al_sgd(
    params: List[Tensor],
    local_weights: List[Tensor],
    dual_weights: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    mu: float,
) -> None:
    """ """
    return al_vr_sgd(
        params,
        local_weights,
        dual_weights,
        None,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        mu=mu,
    )


def mac_sgd(
    params: List[Tensor],
    local_weights: List[Tensor],
    variance_buffer: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    lam: float,
) -> None:
    r"""
    The function that executes the maximizing correlation (Mac) SGD (with variance reduction):
        .. math::
            \DeclareMathOperator*{\argmin}{arg\,min}
            \argmin_x \{f(x) - \lambda \langle x, x_0 \rangle\}

    Parameters
    ----------
    params: list of dict or Parameter,
        the parameters to optimize
    local_weights: iterable of Parameter,
        the (init) local weights,
        i.e. the term `x_0` in
            .. math::
                \lambda \langle x, x_0 \rangle
    variance_buffer: list of Parameter, optional,
        the variance buffers of the parameters,
        used for variance reduction
    d_p_list: list of Tensor,
        the list of gradients of the parameters
    momentum_buffer_list: list of Tensor or list of None,
        the list of momentum buffers,
        works only if `momentum` > 0
    gradient_variance_buffer_list: list of Tensor or list of None,
        the list of gradient variance buffers,
        works only is `vr` is True
    weight_decay: float,
        weight decay (L2 penalty)
    momentum: float,
        momentum factor
    lr: float, default 1e-3,
        the learning rate
    dampening: float,
        dampening for momentum
    nesterov: bool,
        if True, enables Nesterov momentum
    lam: float,
        the (penalty) coeff. of the maximizing correlation term,
        i.e. the term `\lambda` in
            .. math::
                \lambda \langle x, x_0 \rangle

    """
    if local_weights is None:
        local_weights = [None] * len(params)
    if variance_buffer is None:
        variance_buffer = [None] * len(params)
    for idx, (param, lw, vb) in enumerate(zip(params, local_weights, variance_buffer)):
        d_p = d_p_list[idx]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)  # L2 regularization

        if lam != 0 and lw is not None:
            d_p = d_p.sub(lw.detach().clone(), alpha=lam)  # maximizing correlation term

        if momentum != 0:
            buf = momentum_buffer_list[idx]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[idx] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if vb is not None:
            d_p = d_p.sub(vb.detach().clone())  # variance reduction

        param.add_(d_p, alpha=-lr)
