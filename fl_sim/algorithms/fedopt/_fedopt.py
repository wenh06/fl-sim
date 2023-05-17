"""
"""

import warnings
from copy import deepcopy
from typing import List, Sequence

import torch
from tqdm.auto import tqdm

from ...nodes import (
    Server,
    Client,
    ServerConfig,
    ClientConfig,
    ClientMessage,
)


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


class FedOptServerConfig(ServerConfig):

    __name__ = "FedOptServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        optimizer: str = "Adam",
        lr: float = 1e-2,
        betas: Sequence[float] = (0.9, 0.999),
        tau: float = 1e-5,
    ) -> None:
        """
        tau: controls the degree of adaptivity of the algorithm
        """
        assert optimizer.lower() in [
            "avg",
            "adagrad",
            "yogi",
            "adam",
        ]
        super().__init__(
            "FedOpt",
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer=optimizer,
            lr=lr,
            betas=betas,
            tau=tau,
        )


class FedOptClientConfig(ClientConfig):

    __name__ = "FedOptClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        optimizer: str = "SGD",
        **kwargs,
    ) -> None:
        """ """
        super().__init__(
            "FedOpt",
            optimizer,
            batch_size,
            num_epochs,
            lr,
            **kwargs,
        )


class FedOptServer(Server):
    """ """

    __name__ = "FedOptServer"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        self.delta_parameters = self.get_detached_model_parameters()
        for p in self.delta_parameters:
            p.data.zero_()
        if self.config.optimizer.lower() != "avg":
            self.v_parameters = deepcopy(self.delta_parameters)
            for p in self.v_parameters:
                p.data.random_(1, 100).mul_(self.config.tau**2)
        else:
            self.config.lr = 1
            self.config.betas = (1, 0)
            self.v_parameters = [
                # this makes the denominator of the update of the model parameters always 1
                # ref. the last part of self.update
                torch.Tensor([1 - self.config.tau])
                .pow(2)
                .to(self.model.dtype)
                .to(self.device)
                for _ in self.delta_parameters
            ]

    @property
    def client_cls(self) -> "Client":
        return FedOptClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "optimizer",
            "lr",
            "betas",
            "tau",
        ]

    def communicate(self, target: "FedOptClient") -> None:
        """ """
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        """ """
        # update delta_parameters, FedOpt paper Algorithm 2, line 10
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for idx, param in enumerate(self.delta_parameters):
            param.data.mul_(self.config.betas[0])
            for m in self._received_messages:
                param.data.add_(
                    m["delta_parameters"][idx].data.detach().clone().to(self.device),
                    alpha=(1 - self.config.betas[0])
                    * m["train_samples"]
                    / total_samples,
                )
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
        for sp, dp, vp in zip(
            self.model.parameters(), self.delta_parameters, self.v_parameters
        ):
            sp.data.addcdiv_(
                dp.data,
                vp.sqrt() + self.config.tau,
                value=-self.config.lr,
            )

    def update_avg(self) -> None:
        """ """
        pass

    def update_adagrad(self) -> None:
        """ """
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.add_(dp.data.pow(2))

    def update_yogi(self) -> None:
        """ """
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.addcmul_(
                dp.data.pow(2),
                (vp.data - dp.data.pow(2)).sign(),
                value=-(1 - self.config.betas[1]),
            )

    def update_adam(self) -> None:
        """ """
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.mul_(self.config.betas[1]).add_(
                dp.data.pow(2), alpha=1 - self.config.betas[1]
            )

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2003.00295"]


class FedOptClient(Client):

    __name__ = "FedOptClient"

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "optimizer",
        ]

    def communicate(self, target: "FedOptServer") -> None:
        """ """
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
        """ """
        try:
            self.set_parameters(self._received_messages["parameters"])
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        self.solve_inner()  # alias of self.train()

    def train(self) -> None:
        """ """
        self.model.train()
        with tqdm(
            range(self.config.num_epochs), total=self.config.num_epochs, mininterval=1.0
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


class FedAvgServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedAvgServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
    ) -> None:
        """ """
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Avg",
            lr=1,
            betas=(1, 0),
        )


class FedAvgClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedAvgClientConfig"


class FedAvgServer(FedOptServer):
    """ """

    __name__ = "FedAvgServer"

    @property
    def client_cls(self) -> "Client":
        return FedAvgClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return []


class FedAvgClient(FedOptClient):
    """ """

    __name__ = "FedAvgClient"


class FedAdagradServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedAdagradServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float = 1e-2,
        betas: Sequence[float] = (0.9, 0.999),
        tau: float = 1e-5,
    ) -> None:
        """
        tau: controls the degree of adaptivity of the algorithm
        """
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Adagrad",
            lr=lr,
            betas=betas,
            tau=tau,
        )


class FedAdagradClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedAdagradClientConfig"


class FedAdagradServer(FedOptServer):
    """ """

    __name__ = "FedAdagradServer"

    @property
    def client_cls(self) -> "Client":
        return FedAdagradClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [k for k in super().required_config_fields if k != "optimizer"]


class FedAdagradClient(FedOptClient):
    """ """

    __name__ = "FedAdagradClient"


class FedYogiServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedYogiServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float = 1e-2,
        betas: Sequence[float] = (0.9, 0.999),
        tau: float = 1e-5,
    ) -> None:
        """
        tau: controls the degree of adaptivity of the algorithm
        """
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Yogi",
            lr=lr,
            betas=betas,
            tau=tau,
        )


class FedYogiClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedYogiClientConfig"


class FedYogiServer(FedOptServer):
    """ """

    __name__ = "FedYogiServer"

    @property
    def client_cls(self) -> "Client":
        return FedYogiClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [k for k in super().required_config_fields if k != "optimizer"]


class FedYogiClient(FedOptClient):
    """ """

    __name__ = "FedYogiClient"


class FedAdamServerConfig(FedOptServerConfig):
    """ """

    __name__ = "FedAdamServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        lr: float = 1e-2,
        betas: Sequence[float] = (0.9, 0.999),
        tau: float = 1e-5,
    ) -> None:
        """
        tau: controls the degree of adaptivity of the algorithm
        """
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio,
            optimizer="Adam",
            lr=lr,
            betas=betas,
            tau=tau,
        )


class FedAdamClientConfig(FedOptClientConfig):
    """ """

    __name__ = "FedAdamClientConfig"


class FedAdamServer(FedOptServer):
    """ """

    __name__ = "FedAdamServer"

    @property
    def client_cls(self) -> "Client":
        return FedAdamClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [k for k in super().required_config_fields if k != "optimizer"]


class FedAdamClient(FedOptClient):
    """ """

    __name__ = "FedAdamClient"
