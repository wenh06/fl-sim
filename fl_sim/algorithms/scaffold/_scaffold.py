"""
SCAFFOLD re-implemented in the new framework
"""

import warnings
from copy import deepcopy
from typing import List, Any, Dict

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from .._register import _register_algorithm


__all__ = [
    "SCAFFOLDServer",
    "SCAFFOLDClient",
    "SCAFFOLDServerConfig",
    "SCAFFOLDClientConfig",
]


@_register_algorithm("SCAFFOLD")
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
    client_size_aware : bool, default False
        Whether to use client size aware model aggregation.
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``log_dir`` : str or Path, optional
            The log directory.
            If not specified, will use the default log directory.
            If not absolute, will be relative to the default log directory.
        - ``txt_logger`` : bool, default True
            Whether to use txt logger.
        - ``json_logger`` : bool, default True
            Whether to use json logger.
        - ``eval_every`` : int, default 1
            The number of iterations to evaluate the model.
        - ``visiable_gpus`` : Sequence[int], optional
            Visable GPU IDs for allocating devices for clients.
            Defaults to use all GPUs if available.
        - ``seed`` : int, default 0
            The random seed.
        - ``tag`` : str, optional
            The tag of the experiment.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``gpu_proportion`` : float, default 0.2
            The proportion of clients to use GPU.
            Used to similate the system heterogeneity of the clients.
            Not used in the current version, reserved for future use.

    """

    __name__ = "SCAFFOLDServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float,
        client_size_aware: bool = False,
        vr: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "SCAFFOLD",
            num_iters,
            num_clients,
            clients_sample_ratio,
            lr=lr,
            client_size_aware=client_size_aware,
            vr=vr,
        )


@_register_algorithm("SCAFFOLD")
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
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``scheduler`` : dict, optional
            The scheduler config.
            None for no scheduler, using constant learning rate.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``latency`` : float, default 0.0
            The latency of the client.
            Not used in the current version, reserved for future use.

    """

    __name__ = "SCAFFOLDClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        control_variate_update_rule: int = 1,
        vr: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                "The `algorithm` argument fixed to `SCAFFOLD`.", RuntimeWarning
            )
        if kwargs.pop("optimizer", None) is not None:
            warnings.warn(
                "The `optimizer` argument fixed to `SCAFFOLD`.", RuntimeWarning
            )
        super().__init__(
            "SCAFFOLD",
            "SCAFFOLD",
            batch_size,
            num_epochs,
            lr,
            control_variate_update_rule=control_variate_update_rule,
            vr=vr,
            **kwargs,
        )


@_register_algorithm("SCAFFOLD")
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
        assert self.config.vr == self._client_config.vr
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
        if target.config.vr:
            target._received_messages["gradients"] = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in target.model.parameters()
            ]

    def update(self) -> None:
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for m in self._received_messages:
            # update global model parameters
            if self.config.client_size_aware:
                ratio = m["train_samples"] / total_samples
            else:
                # the ratio in the original paper
                ratio = 1.0 / len(self._received_messages)
            self.add_parameters(
                m["parameters_delta"],
                self.config.lr * ratio,
            )
            # update server-side control variates
            ratio *= len(self._received_messages) / len(self._clients)
            for cv, cv_m in zip(self._control_variates, m["control_variates_delta"]):
                cv.add_(cv_m.clone().to(self.device), alpha=ratio)
        if self.config.vr:
            self.update_gradients()

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": SCAFFOLDServerConfig,
            "client": SCAFFOLDClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.1910.06378"]


@_register_algorithm("SCAFFOLD")
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
        self._server_control_variates = [
            torch.zeros_like(p) for p in self.model.parameters()
        ]
        self._updated_control_variates = None  # c_i^+ in the paper
        if self.config.vr:
            self._gradient_buffer = [
                torch.zeros_like(p) for p in self.model.parameters()
            ]
        else:
            self._gradient_buffer = None

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "SCAFFOLDServer") -> None:
        message = {
            "client_id": self.client_id,
            "parameters_delta": [
                mp.sub(cp)
                for mp, cp in zip(
                    self.get_detached_model_parameters(), self._cached_parameters
                )
            ],
            "control_variates_delta": [
                ucv.sub(cv)
                for ucv, cv in zip(
                    self._updated_control_variates, self._control_variates
                )
            ],
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        if self.config.vr:
            message["gradients"] = [
                p.grad.detach().clone() for p in self.model.parameters()
            ]
        target._received_messages.append(ClientMessage(**message))
        for cv, ucv in zip(self._control_variates, self._updated_control_variates):
            cv.copy_(ucv)
        del self._updated_control_variates  # free memory
        self._updated_control_variates = None

    def update(self) -> None:
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn(
                "No parameters received from server. "
                "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        try:
            self._server_control_variates = deepcopy(
                self._received_messages["control_variates"]
            )
        except KeyError:
            warnings.warn(
                "No control variates received from server. "
                "Using current cached server control variates as initial control variates.",
                RuntimeWarning,
            )
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        self._server_control_variates = [
            scv.to(self.device) for scv in self._server_control_variates
        ]
        if (
            self.config.vr
            and self._received_messages.get("gradients", None) is not None
        ):
            self._gradient_buffer = [
                gd.clone().to(self.device)
                for gd in self._received_messages["gradients"]
            ]
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
                    variance_buffer = [
                        scv.detach().clone().sub(cv.detach().clone())
                        for scv, cv in zip(
                            self._server_control_variates, self._control_variates
                        )
                    ]
                    if self._gradient_buffer is not None:
                        for cv, gd in zip(variance_buffer, self._gradient_buffer):
                            cv.add_(gd.clone())
                    self.optimizer.step(
                        variance_buffer=variance_buffer,
                    )
                    # free memory
                    del loss, output, X, y

        # update local control variates
        if self.config.control_variate_update_rule == 1:
            # an additional pass over the local data to compute the gradient at the server model
            # grad_at_server_parameters = [
            #     torch.zeros_like(p) for p in self.model.parameters()
            # ]
            tmp_parameters = [
                p.clone() for p in self.model.parameters()
            ]  # current model parameters
            self.set_parameters(self._cached_parameters)
            self.model.zero_grad()
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                # gradients are accumulated by default
                # for gs, p in zip(grad_at_server_parameters, self.model.parameters()):
                #     gs.add_(
                #         p.grad.detach().clone(),
                #         alpha=len(X) / len(self.train_loader.dataset),
                #     )
            grad_at_server_parameters = [
                p.grad.detach().clone() for p in self.model.parameters()
            ]
            self._updated_control_variates = grad_at_server_parameters
            # recover the current model parameters
            self.set_parameters(tmp_parameters)
            del tmp_parameters
        elif self.config.control_variate_update_rule == 2:
            self._updated_control_variates = deepcopy(self._control_variates)
            for ucv, scv, cp, mp in zip(
                self._updated_control_variates,
                self._server_control_variates,
                self._cached_parameters,
                self.get_detached_model_parameters(),
            ):
                ucv.sub_(scv.detach().clone())
                ucv.add_(
                    cp.detach().clone().sub(mp.detach().clone()),
                    alpha=1.0 / self.config.num_epochs / self.config.lr,
                )
        self.lr_scheduler.step()
