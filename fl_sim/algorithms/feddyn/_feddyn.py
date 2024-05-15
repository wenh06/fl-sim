"""
FedDyn re-implemented in the new framework
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...data_processing.fed_dataset import FedDataset
from ...nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm

__all__ = [
    "FedDynServerConfig",
    "FedDynClientConfig",
    "FedDynServer",
    "FedDynClient",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class FedDynServerConfig(ServerConfig):
    """Server config for the FedDyn algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    mu : float, default 1 / 10
        The coefficient of the "proximal" term.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedDynServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        mu: float = 1 / 10,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ServerConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            name,
            num_iters,
            num_clients,
            clients_sample_ratio,
            mu=mu,
            prox=mu,  # for the `ProxSGD` optimizer
            **kwargs,
        )


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class FedDynClientConfig(ClientConfig):
    """Client config for the FedDyn algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedDynClientConfig"

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
        optimizer = "ProxSGD"
        if kwargs.pop("optimizer", None) is not None:
            warnings.warn(
                "The `optimizer` argument is fixed to `ProxSGD` and will be ignored.",
                RuntimeWarning,
            )
        if kwargs.pop("mu", None) is not None:
            warnings.warn(
                "The `mu` argument of the client will be assigned by the server.",
                RuntimeWarning,
            )
        super().__init__(
            name,
            optimizer,
            batch_size,
            num_epochs,
            lr,
            **kwargs,
        )


@register_algorithm()
@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.",
        "Server node for the FedDyn algorithm.",
    )
    .replace("ServerConfig", "FedDynServerConfig")
    .replace("ClientConfig", "FedDynClientConfig")
)
class FedDynServer(Server):
    """Server node for the FedDyn algorithm."""

    __name__ = "FedDynServer"

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: FedDataset,
        config: FedDynServerConfig,
        client_config: FedDynClientConfig,
        lazy: bool = False,
    ) -> None:
        # assign communication pattern to client config
        setattr(client_config, "mu", config.mu)
        setattr(client_config, "prox", config.prox)
        super().__init__(model, dataset, config, client_config, lazy=lazy)

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        check compatibility of server and client configs,
        and add variables to maintain communication pattern
        """
        super()._post_init()
        self.h_params = [torch.zeros_like(p) for p in self.model.parameters()]

    @property
    def required_config_fields(self) -> List[str]:
        return ["mu"]

    @property
    def client_cls(self) -> type:
        return FedDynClient

    def communicate(self, target: "FedDynClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        """Update the server model and intermidiate variables."""
        # update h_params
        for m in self._received_messages:
            for hp, p, mp in zip(self.h_params, self.model.parameters(), m["parameters"]):
                hp.add_(
                    mp.to(self.device) - p.to(self.device),
                    alpha=-self.config.mu / self.config.num_clients,
                )
        # update global model
        self.avg_parameters()
        for p, hp in zip(self.model.parameters(), self.h_params):
            p = p.add(hp.to(self.device), alpha=-1 / self.config.mu)

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedDynServerConfig,
            "client": FedDynClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/arXiv.2111.04263"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the FedDyn algorithm.",
    ).replace("ClientConfig", "FedDynClientConfig")
)
class FedDynClient(Client):
    """Client node for the FedDyn algorithm."""

    __name__ = "FedDynClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes maintaining the communication pattern
        """
        super()._post_init()
        self.gradients = [torch.zeros_like(p) for p in self.model.parameters()]

    @property
    def required_config_fields(self) -> List[str]:
        return ["mu"]

    def communicate(self, target: "FedDynServer") -> None:
        message = {
            "client_id": self.client_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
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
        self.solve_inner()  # alias of self.train()
        # update local gradients
        for g, p, cp in zip(self.gradients, self.model.parameters(), self._cached_parameters):
            g.add_(p.to(self.device) - cp.to(self.device), alpha=-self.config.mu)

    def train(self) -> None:
        self.model.train()
        local_weights = [p.detach().clone() for p in self._cached_parameters]
        for g, p in zip(self.gradients, local_weights):
            p.add_(g.to(self.device), alpha=1 / self.config.mu)
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
                        local_weights=local_weights,
                    )
                    # free memory
                    # del X, y, output, loss
        self.lr_scheduler.step()
