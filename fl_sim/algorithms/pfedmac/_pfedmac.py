"""
"""

import warnings
from copy import deepcopy
from typing import List

import torch
from tqdm.auto import tqdm

from ...nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage


class pFedMacServerConfig(ServerConfig):
    """ """

    __name__ = "pFedMacServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        beta: float = 1.0,
        vr: bool = False,
    ) -> None:
        """ """
        super().__init__(
            "pFedMac",
            num_iters,
            num_clients,
            clients_sample_ratio,
            beta=beta,
            vr=vr,
        )


class pFedMacClientConfig(ClientConfig):
    """ """

    __name__ = "pFedMacClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        lam: float = 15.0,
        vr: bool = False,
    ) -> None:
        """ """
        super().__init__(
            "pFedMac",
            "pFedMac",
            batch_size,
            num_epochs,
            lr,
            lam=lam,
            vr=vr,
        )


class pFedMacServer(Server):
    """ """

    __name__ = "pFedMacServer"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and check compatibility of server and client configs
        """
        super()._post_init()
        assert self.config.vr == self._client_config.vr

    @property
    def client_cls(self) -> "Client":
        return pFedMacClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return ["beta"]

    def communicate(self, target: "pFedMacClient") -> None:
        """ """
        target._received_messages = {"parameters": self.get_detached_model_parameters()}
        if target.config.vr:
            target._received_messages["gradients"] = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in target.model.parameters()
            ]

    def update(self) -> None:
        """ """
        # sum of received parameters, with self.model.parameters() as its container
        self.avg_parameters(inertia=1 - self.config.beta)
        if self.config.vr:
            self.update_gradients()

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2107.05330"]


class pFedMacClient(Client):
    """ """

    __name__ = "pFedMacClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        if self.config.vr:
            self._gradient_buffer = [
                torch.zeros_like(p) for p in self.model.parameters()
            ]
        else:
            self._gradient_buffer = None

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "lam",
        ]

    def communicate(self, target: "pFedMacServer") -> None:
        """ """
        message = {
            "client_id": self.client_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        if self.config.vr:
            message["gradients"] = [
                p.grad.detach().clone() for p in self.model.parameters()
            ]
        target._received_messages.append(ClientMessage(**message))

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
                        local_weights=self._cached_parameters,
                        variance_buffer=self._gradient_buffer,
                    )
