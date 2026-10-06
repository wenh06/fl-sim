"""
FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction.

`FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction.
<https://arxiv.org/abs/2203.11751>`_ (CVPR 2022)

Codebase URL: https://github.com/GaoLiangFDU/FedDC

"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import ClientMessage
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm
from ..fedopt import FedAvgClient as BaseClient
from ..fedopt import FedAvgClientConfig as BaseClientConfig
from ..fedopt import FedAvgServer as BaseServer
from ..fedopt import FedAvgServerConfig as BaseServerConfig

__all__ = [
    "FedDCClient",
    "FedDCClientConfig",
    "FedDCServer",
    "FedDCServerConfig",
]


_base_algorithm = "FedAvg"


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class FedDCServerConfig(BaseServerConfig):
    """Server config for the FedDC algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float, default 1
        The ratio of clients to participate in each round.
    alpha : float, default 0.01
        The coefficient :math:`\\alpha` of the parameter correction
        (penalized) term :math:`\\frac{\\alpha}{2} \\lVert \\theta_i - (w - h_i) \\rVert^2`.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedDCServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float = 1,
        alpha: float = 0.01,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ServerConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio=clients_sample_ratio,
            **kwargs,
        )
        self.algorithm = name
        self.alpha = alpha


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class FedDCClientConfig(BaseClientConfig):
    """Client config for the FedDC algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of (local) epochs.
    lr : float, default 1e-2
        The learning rate :math:`\\eta`.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedDCClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ClientConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            **kwargs,
        )
        self.algorithm = name


@register_algorithm()
@add_docstring(BaseServer.__doc__.replace(_base_algorithm, "FedDC"))
class FedDCServer(BaseServer):
    __name__ = "FedDCServer"

    def _post_init(self) -> None:
        """Check the configs and initialize the auxiliary variables."""
        super()._post_init()
        assert (
            self.config.alpha >= 0
        ), f"`alpha` should be non-negative, but got {self.config.alpha}."
        # the average of all clients' local update values in the last round,
        # i.e. :math:`g = \\mathbb{E}_{i\\in[N]} g_i` in the paper
        self._global_update = [torch.zeros_like(p) for p in self.model.parameters()]
        # each client's last local update value (gradient drift) :math:`g_i`
        self._client_updates = {
            client_id: [torch.zeros_like(p) for p in self.model.parameters()]
            for client_id in range(self.config.num_clients)
        }

    @property
    def client_cls(self) -> type:
        return FedDCClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedDCServerConfig,
            "client": FedDCClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "FedDCClient") -> None:
        target._received_messages = {
            "parameters": deepcopy(
                [p.detach().clone() for p in self.model.parameters()]
            ),
            "global_update": deepcopy(
                [p.detach().clone() for p in self._global_update]
            ),
            "alpha": self.config.alpha,
        }

    @torch.no_grad()
    def update(self) -> None:
        """Aggregate the drift-corrected local models
        :math:`w = \\sum_i \\frac{|D_i|}{\\sum_j |D_j|} (\\theta_i + h_i)`,
        and update the global gradient drift :math:`g`.
        """
        total_samples = sum(m["train_samples"] for m in self._received_messages)
        aggregated = [torch.zeros_like(p) for p in self.model.parameters()]
        for m in self._received_messages:
            ratio = m["train_samples"] / total_samples
            for ap, p in zip(aggregated, m["parameters"]):
                ap += ratio * p.detach().clone().to(self.device)
            # each client's local update value (gradient drift) of this round
            self._client_updates[m["client_id"]] = [
                u.detach().clone().to(self.device) for u in m["local_update"]
            ]
        for p, ap in zip(self.model.parameters(), aggregated):
            p.data.copy_(ap.data)
        # g = average of all clients' (including inactive ones') last local update values
        self._global_update = [
            torch.mean(
                torch.stack(
                    [self._client_updates[i][k] for i in range(self.config.num_clients)]
                ),
                dim=0,
            )
            for k in range(len(self._global_update))
        ]


@register_algorithm()
@add_docstring(BaseClient.__doc__.replace(_base_algorithm, "FedDC"))
class FedDCClient(BaseClient):
    __name__ = "FedDCClient"

    def _post_init(self) -> None:
        """Check if all required fields in the config are set,
        and set attributes for maintaining intermediate states.
        """
        super()._post_init()
        # local drift variable :math:`h_i`
        self._drift: List[torch.Tensor] = []
        # this client's last local update value :math:`g_i`
        self._local_update: List[torch.Tensor] = []
        # the global average of all clients' local update values :math:`g`
        self._global_update: List[torch.Tensor] = []
        # the coefficient of the parameter correction term, sent by the server
        self._alpha = 0.01

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "FedDCServer") -> None:
        message = {
            "client_id": self.client_id,
            # drift-corrected local model parameters (theta_i + h_i)
            "parameters": self.get_detached_model_parameters(),
            # this round's local update value (gradient drift) delta_theta_i = theta_i^+ - w
            "local_update": self._local_update,
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        w = self._received_messages["parameters"]
        self._global_update = [
            p.to(self.device) for p in self._received_messages["global_update"]
        ]
        self._alpha = self._received_messages.get("alpha", self._alpha)
        if len(self._drift) == 0:  # first participation: h_i = w - w = 0
            self._drift = [torch.zeros_like(p, device=self.device) for p in w]
        if len(self._local_update) == 0:
            self._local_update = [torch.zeros_like(p, device=self.device) for p in w]
        self._cached_parameters = [p.to(self.device) for p in w]
        self.set_parameters(w)
        self.solve_inner()  # alias of self.train()
        # after local training: delta_theta_i = theta_i^+ - w, h_i = h_i + delta_theta_i
        local_params = self.get_detached_model_parameters()
        self._local_update = [
            p.detach().clone() - wp
            for p, wp in zip(local_params, self._cached_parameters)
        ]
        self._drift = [h + d for h, d in zip(self._drift, self._local_update)]

    def train(self) -> None:
        """Local (stochastic) gradient descent on the objective
        :math:`F(\\theta_i) = L_i(\\theta_i) + \\frac{\\alpha}{2} \\lVert \\theta_i - (w - h_i) \\rVert^2
        + \\frac{1}{\\eta K} \\langle \\theta_i, g_i - g \\rangle`,
        where the last term is the gradient (drift) correction,
        cf. Eq. (7) and the local update rule of the paper.
        """
        num_local_batches = self.config.num_epochs * len(self.train_loader)
        # drift correction: (g_i - g) / (eta * K), embedded into the loss as <theta, (g_i - g)> / (eta * K)
        correction = [
            (li - lg) / (self.config.lr * num_local_batches)
            for li, lg in zip(self._local_update, self._global_update)
        ]
        # parameter correction center: w - h_i
        center = [wp - h for wp, h in zip(self._cached_parameters, self._drift)]
        self.model.train()
        with tqdm(
            range(self.config.num_epochs),
            total=self.config.num_epochs,
            mininterval=1.0,
            disable=self.config.verbose < 2,
            leave=False,
        ) as pbar:
            for epoch in pbar:  # local update
                self.model.train()
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    # the penalized (parameter correction) term alpha / 2 * ||theta - (w - h_i)||^2
                    penalty = (
                        self._alpha
                        / 2
                        * sum(
                            ((p - c) ** 2).sum()
                            for p, c in zip(self.model.parameters(), center)
                        )
                    )
                    # the gradient correction (drift) term <theta, (g_i - g)> / (eta * K)
                    correction_term = sum(
                        (p * c).sum()
                        for p, c in zip(self.model.parameters(), correction)
                    )
                    loss = loss + penalty + correction_term
                    loss.backward()
                    self.optimizer.step()
        self.lr_scheduler.step()
