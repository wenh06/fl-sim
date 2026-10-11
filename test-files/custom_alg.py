"""
The same as the FedProx algorithm, for testing custom algorithm.
"""

import warnings
from copy import deepcopy
from typing import Dict, List

import torch
from pydantic import model_validator
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from fl_sim.algorithms._misc import client_config_kw_doc, server_config_kw_doc
from fl_sim.algorithms._register import register_algorithm
from fl_sim.nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig

__all__ = [
    "CustomServer",
    "CustomClient",
    "CustomServerConfig",
    "CustomClientConfig",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class CustomServerConfig(ServerConfig):
    """Server config for the Custom algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "CustomServerConfig"

    vr: bool = False


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class CustomClientConfig(ClientConfig):
    """Client config for the Custom algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    mu : float, default 0.01
        Coefficient for the proximal term.
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "CustomClientConfig"

    lr: float = 1e-2
    mu: float = 0.01
    vr: bool = False
    # NOTE: the optimizer is fixed by the algorithm (derived from `vr`),
    # mirroring ``FedProxClientConfig``
    optimizer: str = ""

    @model_validator(mode="after")
    def _fix_optimizer(self) -> "CustomClientConfig":
        if "optimizer" in self.model_fields_set and self.optimizer:
            warnings.warn(
                "The `optimizer` argument is fixed to `FedProx` or `FedProx_VR` and will be ignored.",
                RuntimeWarning,
            )
        self.optimizer = "FedProx_VR" if self.vr else "FedProx"
        return self


@register_algorithm()
@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.",
        "Server node for the Custom algorithm.",
    )
    .replace("ServerConfig", "CustomServerConfig")
    .replace("ClientConfig", "CustomClientConfig")
)
class CustomServer(Server):
    """Server node for the Custom algorithm."""

    __name__ = "CustomServer"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and check compatibility of server and client configs
        """
        super()._post_init()
        assert self.config.vr == self._client_config.vr

    @property
    def client_cls(self) -> type:
        return CustomClient

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "CustomClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}
        if target.config.vr:
            target._received_messages["gradients"] = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in target.model.parameters()
            ]

    def update(self) -> None:
        # sum of received parameters, with self.model.parameters() as its container
        self.avg_parameters()
        if self.config.vr:
            self.update_gradients()

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": CustomServerConfig,
            "client": CustomClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.1812.06127"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the Custom algorithm.",
    ).replace("ClientConfig", "CustomClientConfig")
)
class CustomClient(Client):
    """Client node for the Custom algorithm."""

    __name__ = "CustomClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        if self.config.vr:
            self._gradient_buffer = [torch.zeros_like(p) for p in self.model.parameters()]
        else:
            self._gradient_buffer = None

    @property
    def required_config_fields(self) -> List[str]:
        return ["mu"]

    def communicate(self, target: "CustomServer") -> None:
        message = {
            "client_id": self.client_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        if self.config.vr:
            message["gradients"] = [p.grad.detach().clone() for p in self.model.parameters()]
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn(
                "No parameters received from server. " "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        if self.config.vr and self._received_messages.get("gradients", None) is not None:
            self._gradient_buffer = [gd.clone().to(self.device) for gd in self._received_messages["gradients"]]
        self.solve_inner()  # alias of self.train()

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
                    )
                    # free memory
                    del X, y, output, loss
        self.lr_scheduler.step()
