"""
"""

from typing import Iterable, Union

from torch.nn.parameter import Parameter

from ._register import register_optimizer
from .base import SGD_VR

__all__ = [
    "SCAFFOLD",
]


@register_optimizer()
class SCAFFOLD(SGD_VR):
    """The ``SCAFFOLD`` optimizer.

    Essentially ``SGD`` with "inertial" (can be implemented as :class:`SGD_VR`),
    i.e., the update rule is

    .. math::

        w_{t+1} = w_t - lr * ( grad(w_t) + control\\_variate )

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
        Model parameters to be optimized.
    lr : float, default 0.01
        Learning rate.

    """

    __name__ = "SCAFFOLD"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
    ) -> None:
        super().__init__(params, lr=lr, momentum=0)
