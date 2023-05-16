"""
"""

import warnings
from copy import deepcopy
from typing import List

from tqdm.auto import tqdm

from ...nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage


__all__ = [
    "FedSplitServer",
    "FedSplitClient",
    "FedSplitServerConfig",
    "FedSplitClientConfig",
]


class FedSplitServerConfig(ServerConfig):
    """ """

    __name__ = "FedSplitServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
    ) -> None:
        """ """
        super().__init__(
            "FedSplit",
            num_iters,
            num_clients,
            clients_sample_ratio,
        )


class FedSplitClientConfig(ClientConfig):
    """ """

    __name__ = "FedSplitClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-3,
        s: float = 10.0,
    ) -> None:
        """ """
        self.s = s
        super().__init__(
            "FedSplit",
            "ProxSGD",
            batch_size,
            num_epochs,
            lr,
            prox=1.0 / s,
        )


class FedSplitServer(Server):
    """ """

    __name__ = "FedSplitServer"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        for c in self._clients:
            # line 2 of Algorithm 1 in the paper
            c._z_parameters = [
                p.to(c.device) for p in self.get_detached_model_parameters()
            ]

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return []

    @property
    def client_cls(self) -> "Client":
        return FedSplitClient

    def communicate(self, target: "FedSplitClient") -> None:
        """ """
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        """ """
        # line 8 of Algorithm 1 in the paper
        self.avg_parameters(size_aware=False)

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2005.05238"]


class FedSplitClient(Client):
    """ """

    __name__ = "FedSplitClient"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        self._z_parameters = None
        # self._z_half_parameters = None  # self.model.parameters() holds this

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return ["s"]

    def communicate(self, target: "FedSplitServer") -> None:
        """ """
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "parameters": [p.clone() for p in self._z_parameters],
                    "train_samples": len(self.train_loader.dataset),
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> None:
        """ """
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        # Local prox step: line 5 of Algorithm 1 in the paper
        self.solve_inner()  # alias of self.train()
        # Local centering step: line 6 of Algorithm 1 in the paper
        for (zp, mp, cp) in zip(
            self._z_parameters, self.model.parameters(), self._cached_parameters
        ):
            zp.add_(mp.detach().clone().sub(cp.detach().clone()), alpha=2.0)

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
                    self.optimizer.step(
                        local_weights=[
                            (2.0 * cp.detach().clone()).sub(zp.detach().clone())
                            for (cp, zp) in zip(
                                self._cached_parameters, self._z_parameters
                            )
                        ]
                    )
