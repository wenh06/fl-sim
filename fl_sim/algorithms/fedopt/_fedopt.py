"""
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Sequence

import torch
from torch_ecg.utils.misc import add_docstring, remove_parameters_returns_from_docstring
from tqdm.auto import tqdm

from ...nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm

__all__ = [
    "FedOptServer",
    "FedOptClient",
    "FedOptServerConfig",
    "FedOptClientConfig",
    "FedAvgServer",
    "FedAvgClient",
    "FedAvgServerConfig",
    "FedAvgClientConfig",
    "FedAdagradServer",
    "FedAdagradClient",
    "FedAdagradServerConfig",
    "FedAdagradClientConfig",
    "FedYogiServer",
    "FedYogiClient",
    "FedYogiServerConfig",
    "FedYogiClientConfig",
    "FedAdamServer",
    "FedAdamClient",
    "FedAdamServerConfig",
    "FedAdamClientConfig",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class FedOptServerConfig(ServerConfig):
    """Server config for the FedOpt algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    optimizer : {"SGD", "Adam", "Adagrad", "Yogi"}, default "Adam"
        The optimizer to use, case insensitive.
    lr : float, default 1e-2
        The learning rate.
    betas : Sequence[float], default (0.9, 0.99)
        The beta values for the optimizer.
    tau : float, default 1e-5
        The tau value for the optimizer,
        which controls the degree of adaptivity of the algorithm.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedOptServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        optimizer: str = "Adam",
        lr: float = 1e-2,
        betas: Sequence[float] = (0.9, 0.99),
        tau: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        assert optimizer.lower() in [
            "avg",
            "adagrad",
            "yogi",
            "adam",
        ], f"Unsupported optimizer: {optimizer}."
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
            optimizer=optimizer,
            lr=lr,
            betas=betas,
            tau=tau,
            **kwargs,
        )


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class FedOptClientConfig(ClientConfig):
    """Client config for the FedOpt algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    optimizer : str, default "SGD"
        The name of the optimizer to solve the local (inner) problem.
    **kwargs : dict, optional
        Additional keyword arguments for specific algorithms
        (FedAvg, FedAdagrad, FedYogi, FedAdam). And
    """

    __name__ = "FedOptClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        optimizer: str = "SGD",
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ClientConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
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
        "Server node for the FedOpt algorithm.",
    )
    .replace("ServerConfig", "FedOptServerConfig")
    .replace("ClientConfig", "FedOptClientConfig")
)
class FedOptServer(Server):
    """Server node for the FedOpt algorithm."""

    __name__ = "FedOptServer"

    def _post_init(self) -> None:
        super()._post_init()
        self.delta_parameters = [torch.zeros_like(p) for p in self.get_detached_model_parameters()]
        if self.config.optimizer.lower() != "avg":
            self.v_parameters = [p.clone() for p in self.delta_parameters]
            for p in self.v_parameters:
                # initialize v_parameters, >= \tau^2
                # FedOpt paper Algorithm 2, line 1
                p.data.random_(1, 100).mul_(self.config.tau**2)
        else:  # FedAvg
            # ensure that the unnecessary parameters are
            # set correctly for the algorithm "FedAvg"
            self.config.lr = 1
            # betas[1] can be set arbitrarily since it is not used for "FedAvg"
            # betas[0] should be set to 0 since "FedAvg" uses no momentum
            self.config.betas = (0, 1)
            # tau can be set arbitrarily since it is not used for "FedAvg"
            self.config.tau = 1
            # set v_parameters to None to avoid unnecessary computation
            self.v_parameters = None

    @property
    def client_cls(self) -> type:
        return FedOptClient

    @property
    def required_config_fields(self) -> List[str]:
        return ["optimizer", "lr", "betas", "tau"]

    def communicate(self, target: "FedOptClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        """Global (outer) update."""
        # update delta_parameters, FedOpt paper Algorithm 2, line 10
        # self._logger_manager.log_message(
        #     f"Before line 10: delta_parameters norm = {FedOptServer.get_norm(self.delta_parameters)}"
        # )
        for idx, param in enumerate(self.delta_parameters):
            param.data.mul_(self.config.betas[0])
            for m in self._received_messages:
                param.data.add_(
                    m["delta_parameters"][idx].data.detach().clone().to(self.device),
                    alpha=(1 - self.config.betas[0]) / len(self._received_messages),
                )
        # self._logger_manager.log_message(
        #     f"After line 10: delta_parameters norm = {FedOptServer.get_norm(self.delta_parameters)}"
        # )
        # update v_parameters, FedOpt paper Algorithm 2, line 11-13
        optimizer = self.config.optimizer.lower()
        if optimizer == "avg":
            self.update_avg()
        elif optimizer == "adagrad":
            self.update_adagrad()
        elif optimizer == "yogi":
            self.update_yogi()
        elif optimizer == "adam":
            self.update_adam()
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        # update model parameters, FedOpt paper Algorithm 2, line 14
        # self._logger_manager.log_message(
        #     f"Before line 14, parameters norm = {FedOptServer.get_norm(self.get_detached_model_parameters())}"
        # )
        if self.v_parameters is None:
            for sp, dp in zip(self.model.parameters(), self.delta_parameters):
                sp.data.add_(dp.data, alpha=self.config.lr)
        else:
            for sp, dp, vp in zip(self.model.parameters(), self.delta_parameters, self.v_parameters):
                sp.data.addcdiv_(
                    dp.data,
                    vp.sqrt() + self.config.tau,
                    value=self.config.lr,
                )
        # self._logger_manager.log_message(
        #     f"After line 14, parameters norm = {FedOptServer.get_norm(self.get_detached_model_parameters())}"
        # )

    def update_avg(self) -> None:
        """Additional operation for FedAvg."""
        # do nothing
        # FedAvg does not use delta_parameters nor v_parameters
        pass

    def update_adagrad(self) -> None:
        """Additional operation for FedAdagrad."""
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.add_(dp.data.pow(2))

    def update_yogi(self) -> None:
        """Additional operation for FedYogi."""
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.addcmul_(
                dp.data.pow(2),
                (vp.data - dp.data.pow(2)).sign(),
                value=-(1 - self.config.betas[1]),
            )

    def update_adam(self) -> None:
        """Additional operation for FedAdam."""
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.mul_(self.config.betas[1]).add_(dp.data.pow(2), alpha=1 - self.config.betas[1])

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedOptServerConfig,
            "client": FedOptClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2003.00295"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the FedOpt algorithm.",
    ).replace("ClientConfig", "FedOptClientConfig")
)
class FedOptClient(Client):
    """Client node for the FedOpt algorithm."""

    __name__ = "FedOptClient"

    @property
    def required_config_fields(self) -> List[str]:
        return ["optimizer"]

    def communicate(self, target: "FedOptServer") -> None:
        delta_parameters = self.get_detached_model_parameters()
        for dp, rp in zip(delta_parameters, self._cached_parameters):
            dp.data.add_(rp.data, alpha=-1)
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "delta_parameters": delta_parameters,
                    "train_samples": len(self.train_loader.dataset),
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> None:
        try:
            self.set_parameters(self._received_messages["parameters"])
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
                    self.optimizer.step()
                    # free memory
                    # del X, y, output, loss
        self.lr_scheduler.step()


@register_algorithm()
@add_docstring(
    remove_parameters_returns_from_docstring(FedOptServerConfig.__doc__, parameters=["optimizer", "lr", "betas", "tau"])
)
class FedAvgServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedAvgServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("lr", None) is not None:
            warnings.warn(
                "`lr` is fixed to `1` for FedAvgServerConfig and will be ignored.",
                RuntimeWarning,
            )
        if kwargs.pop("betas", None) is not None:
            warnings.warn(
                "`betas` is fixed to `(0, 1)` for FedAvgServerConfig and will be ignored.",
                RuntimeWarning,
            )
        if kwargs.pop("tau", None) is not None:
            warnings.warn(
                "`tau` is not used for FedAvgServerConfig and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Avg",
            lr=1,
            betas=(0, 1),  # betas[1] can be set arbitrarily since it is not used
            tau=1,  # tau can be set arbitrarily since it is not used
            **kwargs,
        )
        self.algorithm = "FedAvg"


@register_algorithm()
@add_docstring(FedOptClientConfig.__doc__.replace("FedOpt", "FedAvg"))
class FedAvgClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedAvgClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        optimizer: str = "SGD",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            optimizer=optimizer,
            **kwargs,
        )
        self.algorithm = "FedAvg"


@register_algorithm()
@add_docstring(FedOptServer.__doc__.replace("FedOpt", "FedAvg"))
class FedAvgServer(FedOptServer):
    """Server node for the FedAvg algorithm."""

    __name__ = "FedAvgServer"

    @property
    def client_cls(self) -> type:
        return FedAvgClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedAvgServerConfig,
            "client": FedAvgClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return []


@register_algorithm()
@add_docstring(FedOptClient.__doc__.replace("FedOpt", "FedAvg"))
class FedAvgClient(FedOptClient):
    """ """

    __name__ = "FedAvgClient"


@register_algorithm()
@add_docstring(remove_parameters_returns_from_docstring(FedOptServerConfig.__doc__, parameters=["optimizer"]))
class FedAdagradServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedAdagradServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float = 1e-2,
        betas: Sequence[float] = (0.0, 0.99),
        tau: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Adagrad",
            lr=lr,
            betas=betas,
            tau=tau,
            **kwargs,
        )
        self.algorithm = "FedAdagrad"


@register_algorithm()
@add_docstring(FedOptClientConfig.__doc__.replace("FedOpt", "FedAdagrad"))
class FedAdagradClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedAdagradClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        optimizer: str = "SGD",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            optimizer=optimizer,
            **kwargs,
        )
        self.algorithm = "FedAdagrad"


@register_algorithm()
@add_docstring(FedOptServer.__doc__.replace("FedOpt", "FedAdagrad"))
class FedAdagradServer(FedOptServer):
    """ """

    __name__ = "FedAdagradServer"

    @property
    def client_cls(self) -> type:
        return FedAdagradClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedAdagradServerConfig,
            "client": FedAdagradClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return [k for k in super().required_config_fields if k != "optimizer"]


@register_algorithm()
@add_docstring(FedOptClient.__doc__.replace("FedOpt", "FedAdagrad"))
class FedAdagradClient(FedOptClient):
    """ """

    __name__ = "FedAdagradClient"


@register_algorithm()
@add_docstring(remove_parameters_returns_from_docstring(FedOptServerConfig.__doc__, parameters=["optimizer"]))
class FedYogiServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedYogiServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float = 1e-2,
        betas: Sequence[float] = (0.9, 0.99),
        tau: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Yogi",
            lr=lr,
            betas=betas,
            tau=tau,
            **kwargs,
        )
        self.algorithm = "FedYogi"


@register_algorithm()
@add_docstring(FedOptClientConfig.__doc__.replace("FedOpt", "FedYogi"))
class FedYogiClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedYogiClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        optimizer: str = "SGD",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            optimizer=optimizer,
            **kwargs,
        )
        self.algorithm = "FedYogi"


@register_algorithm()
@add_docstring(FedOptServer.__doc__.replace("FedOpt", "FedYogi"))
class FedYogiServer(FedOptServer):
    """ """

    __name__ = "FedYogiServer"

    @property
    def client_cls(self) -> type:
        return FedYogiClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedYogiServerConfig,
            "client": FedYogiClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return [k for k in super().required_config_fields if k != "optimizer"]


@register_algorithm()
@add_docstring(FedOptClient.__doc__.replace("FedOpt", "FedYogi"))
class FedYogiClient(FedOptClient):
    """ """

    __name__ = "FedYogiClient"


@register_algorithm()
@add_docstring(remove_parameters_returns_from_docstring(FedOptServerConfig.__doc__, parameters=["optimizer"]))
class FedAdamServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedAdamServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float = 1e-2,
        betas: Sequence[float] = (0.9, 0.99),
        tau: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Adam",
            lr=lr,
            betas=betas,
            tau=tau,
            **kwargs,
        )
        self.algorithm = "FedAdam"


@register_algorithm()
@add_docstring(FedOptClientConfig.__doc__.replace("FedOpt", "FedAdam"))
class FedAdamClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedAdamClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        optimizer: str = "SGD",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            optimizer=optimizer,
            **kwargs,
        )
        self.algorithm = "FedAdam"


@register_algorithm()
@add_docstring(FedOptServer.__doc__.replace("FedOpt", "FedAdam"))
class FedAdamServer(FedOptServer):
    """ """

    __name__ = "FedAdamServer"

    @property
    def client_cls(self) -> type:
        return FedAdamClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedAdamServerConfig,
            "client": FedAdamClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return [k for k in super().required_config_fields if k != "optimizer"]


@register_algorithm()
@add_docstring(FedOptClient.__doc__.replace("FedOpt", "FedAdam"))
class FedAdamClient(FedOptClient):
    """ """

    __name__ = "FedAdamClient"
