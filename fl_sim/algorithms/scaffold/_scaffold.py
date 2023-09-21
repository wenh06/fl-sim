"""
SCAFFOLD re-implemented in the new framework

Warning: possible memory leak in the current version?
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm

__all__ = [
    "SCAFFOLDServer",
    "SCAFFOLDClient",
    "SCAFFOLDServerConfig",
    "SCAFFOLDClientConfig",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class SCAFFOLDServerConfig(ServerConfig):
    """Server config for the SCAFFOLD algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    lr : float
        The learning rate.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "SCAFFOLDServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float,
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
            lr=lr,
            **kwargs,
        )


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class SCAFFOLDClientConfig(ClientConfig):
    """Client config for the SCAFFOLD algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    control_variate_update_rule : int, default 1
        The update rule for the control variates.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "SCAFFOLDClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        control_variate_update_rule: int = 1,
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
            # name,  # parameter explosion observed using the `SGD_VR` optimizer
            "SGD",
            batch_size,
            num_epochs,
            lr,
            control_variate_update_rule=control_variate_update_rule,
            **kwargs,
        )


@register_algorithm()
@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.",
        "Server node for the SCAFFOLD algorithm.",
    )
    .replace("ServerConfig", "SCAFFOLDServerConfig")
    .replace("ClientConfig", "SCAFFOLDClientConfig")
)
class SCAFFOLDServer(Server):
    """Server node for the SCAFFOLD algorithm."""

    __name__ = "SCAFFOLDServer"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and check compatibility of server and client configs
        """
        super()._post_init()
        self._control_variates = [torch.zeros_like(p) for p in self.model.parameters()]

    @property
    def client_cls(self) -> type:
        return SCAFFOLDClient

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "SCAFFOLDClient") -> None:
        target._received_messages = {
            "parameters": self.get_detached_model_parameters(),
            "control_variates": deepcopy(self._control_variates),
        }

    def update(self) -> None:
        # SCAFFOLD paper Algorithm 1 line 16-17
        ratio_p = self.config.lr / len(self._received_messages)
        ratio_c = 1 / len(self._clients)
        for m in self._received_messages:
            # update global model parameters
            self.add_parameters(m["parameters_delta"], ratio_p)
            # update server-side control variates
            for cv, cv_m in zip(self._control_variates, m["control_variates_delta"]):
                cv.add_(cv_m.clone().to(self.device), alpha=ratio_c)

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": SCAFFOLDServerConfig,
            "client": SCAFFOLDClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.1910.06378"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the SCAFFOLD algorithm.",
    ).replace("ClientConfig", "SCAFFOLDClientConfig")
)
class SCAFFOLDClient(Client):
    """Client node for the SCAFFOLD algorithm."""

    __name__ = "SCAFFOLDClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        assert self.config.control_variate_update_rule in [1, 2]
        self._control_variates = [torch.zeros_like(p) for p in self.model.parameters()]
        self._server_control_variates = [torch.zeros_like(p) for p in self.model.parameters()]
        self._updated_control_variates = None  # c_i^+ in the paper

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "SCAFFOLDServer") -> None:
        message = {
            "client_id": self.client_id,
            "parameters_delta": [mp.sub(cp) for mp, cp in zip(self.get_detached_model_parameters(), self._cached_parameters)],
            "control_variates_delta": [ucv.sub(cv) for ucv, cv in zip(self._updated_control_variates, self._control_variates)],
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        target._received_messages.append(ClientMessage(**message))
        for cv, ucv in zip(self._control_variates, self._updated_control_variates):
            cv.copy_(ucv)
        del self._updated_control_variates  # free memory
        self._updated_control_variates = None

    def update(self) -> None:
        try:
            # self.set_parameters(self._received_messages["parameters"])
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn(
                "No parameters received from server. " "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        try:
            self._server_control_variates = deepcopy(self._received_messages["control_variates"])
        except KeyError:
            warnings.warn(
                "No control variates received from server. "
                "Using current cached server control variates as initial control variates.",
                RuntimeWarning,
            )
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        self._server_control_variates = [scv.to(self.device) for scv in self._server_control_variates]
        self.solve_inner()  # alias of self.train()

    def train(self) -> None:
        # SCAFFOLD paper Algorithm 1 line 10
        # c - c_i, control variates, compute in advance
        variance_buffer = [
            scv.detach().clone().sub(cv.detach().clone())
            for scv, cv in zip(self._server_control_variates, self._control_variates)
        ]
        self.model.train()
        mini_batch_grads = [torch.zeros_like(p) for p in self._control_variates]
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
                    # should step or not?
                    self.optimizer.step(
                        # variance_buffer=variance_buffer,
                    )
                    for p, g in zip(self.model.parameters(), mini_batch_grads):
                        g.add_(p.grad.detach().clone(), alpha=1.0 / self.config.num_epochs)
                    # free memory
                    del loss, output, X, y
        for p, g, v in zip(self.model.parameters(), mini_batch_grads, variance_buffer):
            p = p.add(g.detach().clone().add(v.detach().clone()), alpha=-self.config.lr)

        del variance_buffer, mini_batch_grads

        # update local control variates
        # SCAFFOLD paper Algorithm 1 line 12
        if self.config.control_variate_update_rule == 1:
            # an additional pass over the local data to compute the gradient at the server model
            self._updated_control_variates = self.compute_gradients(at=self._cached_parameters)
        elif self.config.control_variate_update_rule == 2:
            self._updated_control_variates = deepcopy(self._control_variates)
            for ucv, scv, cp, mp in zip(
                self._updated_control_variates,
                self._server_control_variates,
                self._cached_parameters,
                self.get_detached_model_parameters(),
            ):
                ucv.sub_(scv.detach().clone()).add_(
                    cp.detach().clone().sub(mp.detach().clone()),
                    alpha=1.0 / self.config.num_epochs / self.config.lr,
                )
        self.lr_scheduler.step()
