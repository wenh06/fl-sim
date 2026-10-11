"""
FedPD re-implemented in the new framework
"""

import warnings
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from pydantic import model_validator
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...data_processing.fed_dataset import FedDataset
from ...nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm

__all__ = [
    "FedPDServerConfig",
    "FedPDClientConfig",
    "FedPDServer",
    "FedPDClient",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class FedPDServerConfig(ServerConfig):
    """Server config for the FedPD algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    p : float
        Probability of skipping communication.
    stochastic : bool, default False
        Skip communication in a stochastic manner or not.
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedPDServerConfig"

    clients_sample_ratio: float = 1
    p: float
    stochastic: bool = False
    vr: bool = False

    @model_validator(mode="after")
    def _set_comm_freq(self) -> "FedPDServerConfig":
        # clients communicate with the server once per `comm_freq` local steps,
        # derived from the sampling ratio `p` (as in the original implementation)
        self.comm_freq = int(1 / self.p)
        return self


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class FedPDClientConfig(ClientConfig):
    """Client config for the FedPD algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    mu : float, default 1 / 10
        The coefficient of the proximal term.
    vr : bool, default False
        Whether to use variance reduction.
    dual_rand_init : bool, default False
        Whether to use random initialization for dual variables.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedPDClientConfig"

    # NOTE: the optimizer is fixed by the algorithm (derived from `vr`);
    # declared optional here and set by the validator below
    optimizer: str = ""

    lr: float = 1e-2
    mu: float = 1 / 10
    vr: bool = False
    dual_rand_init: bool = False

    @model_validator(mode="after")
    def _fix_optimizer(self) -> "FedPDClientConfig":
        if "optimizer" in self.model_fields_set:
            warnings.warn(
                "The `optimizer` argument is fixed to `FedPD_VR` or `FedPD_SGD` and will be ignored.",
                RuntimeWarning,
            )
        self.optimizer = "FedPD_VR" if self.vr else "FedPD_SGD"
        return self


@register_algorithm()
@add_docstring(
    Server.__doc__.replace("The class to simulate the server node.", "Server node for the FedPD algorithm.")
    .replace("ServerConfig", "FedPDServerConfig")
    .replace("ClientConfig", "FedPDClientConfig")
)
class FedPDServer(Server):
    """Server node for the FedPD algorithm."""

    __name__ = "FedPDServer"

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: FedDataset,
        config: FedPDServerConfig,
        client_config: FedPDClientConfig,
        lazy: bool = False,
    ) -> None:
        # assign communication pattern to client config
        setattr(client_config, "p", config.p)
        setattr(client_config, "stochastic", config.stochastic)
        setattr(client_config, "comm_freq", config.comm_freq)
        super().__init__(model, dataset, config, client_config, lazy=lazy)

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        check compatibility of server and client configs,
        and add variables to maintain communication pattern
        """
        super()._post_init()
        assert self.config.vr == self._client_config.vr
        self._communicated_clients = []

    @property
    def client_cls(self) -> type:
        return FedPDClient

    def communicate(self, target: "FedPDClient") -> None:
        if target.client_id not in self._communicated_clients:
            return
        target._received_messages = {
            "parameters": [p.detach().clone() for p in self.model.parameters()],
        }
        if target.config.vr:
            target._received_messages["gradients"] = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in target.model.parameters()
            ]

    def update(self) -> None:
        self._communicated_clients = [m["client_id"] for m in self._received_messages]
        # sum of received parameters, with self.model.parameters() as its container
        self.avg_parameters()
        if self.config.vr:
            self.update_gradients()

    def aggregate_client_metrics(self) -> None:
        """skip aggregation if no client has communicated"""
        if len(self._received_messages) == 0:
            return
        super().aggregate_client_metrics()

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedPDServerConfig,
            "client": FedPDClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.1109/tsp.2021.3115952"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace("The class to simulate the client node.", "Client node for the FedPD algorithm.").replace(
        "ClientConfig", "FedPDClientConfig"
    )
)
class FedPDClient(Client):
    """Client node for the FedPD algorithm."""

    __name__ = "FedPDClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes maintaining the communication pattern
        """
        super()._post_init()
        if self.config.vr:
            self._gradient_buffer = [torch.zeros_like(p) for p in self.model.parameters()]
        else:
            self._gradient_buffer = None
        if self.config.dual_rand_init:
            self._dual_weights = [torch.randn_like(p) for p in self.model.parameters()]
        else:
            self._dual_weights = [torch.zeros_like(p) for p in self.model.parameters()]

    def communicate(self, target: "FedPDServer") -> None:
        # determine if communication happens
        # the probability of communication is controlled by `config.p`
        # the pattern (stochastic or every n iters) is controlled by `config.stochastic`
        if self.config.stochastic:
            if np.random.rand() >= self.config.p:
                # self._communicate will automatically augment
                # target._num_communications by 1
                target._num_communications -= 1
                return
        else:
            if (target.n_iter + 1) % self.config.comm_freq != 0:
                target._num_communications -= 1
                return
        message = {
            "client_id": self.client_id,
            # "parameters": self.get_detached_model_parameters(),
            "parameters": self._cached_parameters,
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        if self.config.vr:
            message["gradients"] = [p.grad.detach().clone() for p in self.model.parameters()]
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        # x_i^r: self.model.parameters()
        # x_{0,i}^r, x_{0,i}^{r+}: self._cached_parameters
        # \lambda_i^r: self._dual_weights
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn(
                "No parameters received from server. " "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
            if self._cached_parameters is None:
                self._cached_parameters = [p.detach().clone() for p in self.model.parameters()]
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        if self.config.vr and self._received_messages.get("gradients", None) is not None:
            self._gradient_buffer = [gd.clone().to(self.device) for gd in self._received_messages["gradients"]]
        self.solve_inner()  # alias of self.train()
        # update dual weights and cached parameters
        for i, p in enumerate(self.model.parameters()):
            self._dual_weights[i].add_(
                p.detach().clone() - self._cached_parameters[i],
                alpha=1.0 / self.config.mu,
            )
            # the x_{0,i}^{r+} in the paper
            self._cached_parameters[i].add_(self._dual_weights[i], alpha=self.config.mu)

    def train(self) -> None:
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
                    loss.backward()
                    self.optimizer.step(
                        local_weights=self._cached_parameters,
                        variance_buffer=self._gradient_buffer,
                        dual_weights=self._dual_weights,
                    )
                    # free memory
                    # del X, y, output, loss
        self.lr_scheduler.step()
