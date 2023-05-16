"""
"""

import warnings
from copy import deepcopy
from typing import List, Optional, Dict

import torch
from tqdm.auto import tqdm

from ...nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from ...optimizers import get_optimizer


__all__ = [
    "DittoServerConfig",
    "DittoClientConfig",
    "DittoServer",
    "DittoClient",
]


class DittoServerConfig(ServerConfig):
    """ """

    __name__ = "DittoServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
    ) -> None:
        """ """
        super().__init__(
            "Ditto",
            num_iters,
            num_clients,
            clients_sample_ratio,
        )


class DittoClientConfig(ClientConfig):
    """ """

    __name__ = "DittoClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        optimizer: str = "ProxSGD",
        optimizer_per: str = "SGD",
        prox: float = 0.01,
        lr: float = 1e-3,
        lr_per: Optional[float] = None,
    ) -> None:
        """ """
        super().__init__(
            "Ditto",
            optimizer,
            batch_size,
            num_epochs,
            lr,
            prox=prox,
            optimizer_per=optimizer_per,
            lr_per=lr_per if lr_per is not None else lr,
        )


class DittoServer(Server):
    """ """

    __name__ = "DittoServer"

    @property
    def client_cls(self) -> "Client":
        return DittoClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return []

    def communicate(self, target: "DittoClient") -> None:
        """ """
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        """ """
        # sum of received parameters, with self.model.parameters() as its container
        self.avg_parameters()

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2012.04221"]


class DittoClient(Client):
    """ """

    __name__ = "DittoClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()

        self.model_per = deepcopy(self.model)

        config_per = {
            "lr": self.config.lr_per,
        }
        self.optimizer_per = get_optimizer(
            optimizer_name=self.config.optimizer_per,
            params=self.model_per.parameters(),
            config=config_per,
        )

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return ["optimizer_per", "lr_per"]

    def communicate(self, target: "DittoServer") -> None:
        """ """
        message = {
            "client_id": self.client_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
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
        self.set_parameters(self._cached_parameters)
        self.solve_inner()  # alias of self.train()
        self.train_per()

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
                    self.optimizer.step(local_weights=self._cached_parameters)

    def train_per(self) -> None:
        """ """
        self.model_per.train()
        with tqdm(
            range(self.config.num_epochs), total=self.config.num_epochs, mininterval=1.0
        ) as pbar:
            for epoch in pbar:  # local update
                self.model_per.train()
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer_per.zero_grad()
                    output = self.model_per(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer_per.step(local_weights=self._cached_parameters)

    @torch.no_grad()
    def evaluate(self, part: str) -> Dict[str, float]:
        """
        evaluate the model and personalized model
        on the given part of the dataset.

        Parameters
        ----------
        part : str,
            The part of the dataset to evaluate on,
            can be either "train" or "val".

        Returns
        -------
        `Dict[str, float]`,
            The metrics of the evaluation.

        """
        assert part in self.dataset.data_parts, "Invalid part name"
        for idx, model in enumerate([self.model, self.model_per]):
            model.eval()
            _metrics = []
            data_loader = self.val_loader if part == "val" else self.train_loader
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = model(X)
                _metrics.append(self.dataset.evaluate(logits, y))
            if part not in self._metrics:
                self._metrics[part] = {
                    "num_samples": sum([m["num_samples"] for m in _metrics]),
                }
            suffix = "_per" if idx == 1 else ""
            for k in _metrics[0]:
                if k != "num_samples":  # average over all metrics
                    self._metrics[part][f"{k}{suffix}"] = (
                        sum([m[k] * m["num_samples"] for m in _metrics])
                        / self._metrics[part]["num_samples"]
                    )
        return self._metrics[part]
