"""
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig
from ...regularizers import get_regularizer
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm

__all__ = [
    "FedDRServer",
    "FedDRClient",
    "FedDRServerConfig",
    "FedDRClientConfig",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class FedDRServerConfig(ServerConfig):
    """Server config for the FedDR algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    eta : float, default 1.0
        The eta (regularization) parameter for the FedDR algorithm.
    alpha : float, default 1.9
        The alpha parameter for the FedDR algorithm.
    reg_type : str, default "l1_norm"
        The type of regularizer to use for the FedDR algorithm.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedDRServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        eta: float = 1.0,
        alpha: float = 1.9,
        reg_type: str = "l1_norm",
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
            eta=eta,
            alpha=alpha,
            reg_type=reg_type,
            **kwargs,
        )


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class FedDRClientConfig(ClientConfig):
    """Client config for the FedDR algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    eta : float, default 1.0
        The eta (regularization) parameter for the FedDR algorithm.
    alpha : float, default 1.9
        The alpha parameter for the FedDR algorithm.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedDRClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        eta: float = 1.0,
        alpha: float = 1.9,  # in the FedDR paper, clients' alpha is equal to the server's alpha
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ClientConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        if kwargs.pop("optimizer", None) is not None:
            warnings.warn(
                f"The `optimizer` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            name,
            name,
            batch_size,
            num_epochs,
            lr,
            eta=eta,
            alpha=alpha,
            **kwargs,
        )


@register_algorithm()
@add_docstring(
    Server.__doc__.replace("The class to simulate the server node.", "Server node for the FedDR algorithm.")
    .replace("ServerConfig", "FedDRServerConfig")
    .replace("ClientConfig", "FedDRClientConfig")
)
class FedDRServer(Server):
    """Server node for the FedDR algorithm."""

    __name__ = "FedDRServer"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        self._regularizer = get_regularizer(
            self.config.reg_type,
            self.config.eta * self.config.num_clients / (self.config.num_clients + 1),
        )
        self._y_parameters = self.get_detached_model_parameters()  # y
        self._x_til_parameters = self.get_detached_model_parameters()  # x_tilde

    @property
    def client_cls(self) -> type:
        return FedDRClient

    @property
    def required_config_fields(self) -> List[str]:
        return ["alpha", "eta", "reg_type"]

    def communicate(self, target: "FedDRClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        """Global (outer) update."""
        # update y
        # FedDR paper Algorithm 1 line 7, first equation
        for yp, mp in zip(self._y_parameters, self.model.parameters()):
            yp.data.add_(mp.data - yp.data, alpha=self.config.alpha)

        # update x_tilde
        # FedDR paper Algorithm 1 line 7, second equation
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for m in self._received_messages:
            for i, xtp in enumerate(self._x_til_parameters):
                xtp.data.add_(
                    m["x_hat_delta"][i].data.to(self.device),
                    alpha=m["train_samples"] / total_samples,
                )

        # update server (global) model
        # FedDR paper Algorithm 1 line 8
        for mp, yp, xtp in zip(self.model.parameters(), self._y_parameters, self._x_til_parameters):
            mp.data = (self._regularizer.coeff / self.config.eta) * xtp.data + (1 / (self.config.num_clients + 1)) * yp.data
        for mp, p in zip(
            self.model.parameters(),
            self._regularizer.prox_eval(params=self.model.parameters()),
        ):
            mp.data = p.data

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedDRServerConfig,
            "client": FedDRClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2103.03452"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace("The class to simulate the client node.", "Client node for the FedDR algorithm.").replace(
        "ClientConfig", "FedDRClientConfig"
    )
)
class FedDRClient(Client):
    """Client node for the FedDR algorithm."""

    __name__ = "FedDRClient"

    def _post_init(self) -> None:
        super()._post_init()
        self._y_parameters = None  # y
        self._x_hat_parameters = None  # x_hat
        self._x_hat_buffer = None  # x_hat_buffer

    @property
    def required_config_fields(self) -> List[str]:
        return ["alpha", "eta"]

    def communicate(self, target: "FedDRServer") -> None:
        if self._x_hat_buffer is None:
            # outter iteration step -1, no need to communicate
            x_hat_delta = [torch.zeros_like(p) for p in self._x_hat_parameters]
        else:
            x_hat_delta = [p.data - hp.data for p, hp in zip(self._x_hat_parameters, self._x_hat_buffer)]
        self._x_hat_buffer = [p.clone() for p in self._x_hat_parameters]
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "x_hat_delta": x_hat_delta,
                    "train_samples": len(self.train_loader.dataset),
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> None:
        """Local (inner) update."""
        # copy the parameters from the server
        # x_bar
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
        # update y
        if self._y_parameters is None:
            self._y_parameters = [p.clone().to(self.device) for p in self._cached_parameters]
        else:
            for yp, cp, mp in zip(self._y_parameters, self._cached_parameters, self.model.parameters()):
                yp.data.add_(cp.data - mp.data, alpha=self.config.alpha)
        # update x, via prox_sgd of y
        self.solve_inner()  # alias of self.train()
        # update x_hat
        if self._x_hat_parameters is None:
            self._x_hat_parameters = [p.clone().to(self.device) for p in self._cached_parameters]
        for hp, yp, mp in zip(self._x_hat_parameters, self._y_parameters, self.model.parameters()):
            hp.data = 2 * mp.data - yp.data

    def train(self) -> None:
        """Train the local model for ``num_epochs`` epochs."""
        self.model.train()
        with tqdm(
            range(self.config.num_epochs),
            total=self.config.num_epochs,
            mininterval=1.0,
            disable=self.config.verbose < 2,
        ) as pbar:
            for epoch in pbar:  # local update
                self.model.train()
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer.step(self._y_parameters)
                    # free memory
                    # del X, y, output, loss
        self.lr_scheduler.step()
