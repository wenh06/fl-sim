"""
For original implementation, see `FedPD <https://github.com/564612540/FedPD>`_.
"""

from typing import Iterable, Union

from torch.nn import Parameter
from torch_ecg.utils import add_docstring

from .base import AL_SGD_VR, AL_SGD
from ._register import register_optimizer


__all__ = [
    "FedPD_VR",
    "FedPD_SGD",
]


@register_optimizer()
class FedPD_VR(AL_SGD_VR):
    """Local optimizer for ``FedPD`` using SGD with variance reduction.

    Parameters
    ----------
    params : Iterable[dict] or Iterable[torch.nn.parameter.Parameter]
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


@register_optimizer()
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
