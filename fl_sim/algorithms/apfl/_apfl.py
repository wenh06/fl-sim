"""
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
    "APFLServerConfig",
    "APFLClientConfig",
    "APFLServer",
    "APFLClient",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class APFLServerConfig(ServerConfig):
    """Server config for the APFL algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    tau : int, default 1,
        synchronization gap.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "APFLServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        tau: int = 1,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ServerConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        assert isinstance(tau, int) and tau >= 1, "`tau` (synchronization gap) must be a positive integer."
        super().__init__(
            name,
            num_iters,
            num_clients,
            clients_sample_ratio,
            tau=tau,
            **kwargs,
        )


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class APFLClientConfig(ClientConfig):
    """Client config for the APFL algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    optimizer : str, default "SGD"
        The name of the optimizer to solve the local (inner) problem.
    lr : float, default 1e-2
        The learning rate.
    mixture_weight : float, default 0.5
        The weight of the local model parameters and the global model parameters.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "APFLClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        optimizer: str = "SGD",
        lr: float = 1e-2,
        mixture_weight: float = 0.5,
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
            mixture_weight=mixture_weight,
            **kwargs,
        )


@register_algorithm()
@add_docstring(
    Server.__doc__.replace("The class to simulate the server node.", "Server node for the APFL algorithm.")
    .replace("ServerConfig", "APFLServerConfig")
    .replace("ClientConfig", "APFLClientConfig")
)
class APFLServer(Server):
    """Server node for the APFL algorithm."""

    __name__ = "APFLServer"

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: FedDataset,
        config: APFLServerConfig,
        client_config: APFLClientConfig,
        lazy: bool = False,
    ) -> None:
        # assign communication pattern to client config
        setattr(client_config, "sync_gap", config.tau)
        super().__init__(model, dataset, config, client_config, lazy=lazy)

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        self._selected_clients = super()._sample_clients()

    def _sample_clients(self) -> List[int]:
        """Sample clients for the current iteration."""
        if self.n_iter % self.config.tau == 0:
            # if the current iteration divides the synchronization gap,
            # update the selected clients
            self._selected_clients = super()._sample_clients()
            # else, keep the selected clients unchanged
        return self._selected_clients

    @property
    def client_cls(self) -> type:
        return APFLClient

    @property
    def required_config_fields(self) -> List[str]:
        return ["tau"]

    def communicate(self, target: "APFLClient") -> None:
        if self.n_iter % self.config.tau == 0:
            # if the current iteration divides the synchronization gap,
            # transmit the global model parameters to the client
            # otherwise, do nothing
            target._received_messages = {"parameters": self.get_detached_model_parameters()}

    @add_docstring(Server.update)
    def update(self) -> None:
        if self.n_iter % self.config.tau == 0:
            # if the current iteration divides the synchronization gap,
            # update the global model parameters via averaging
            # otherwise, do nothing
            # NOTE that the re-sampling of clients is done
            # in `self._sample_clients()` in the training loop
            self.avg_parameters()

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": APFLServerConfig,
            "client": APFLClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2003.13461"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace("The class to simulate the client node.", "Client node for the APFL algorithm.").replace(
        "ClientConfig", "APFLClientConfig"
    )
)
class APFLClient(Client):
    """Client node for the APFL algorithm."""

    __name__ = "APFLClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        self.model_per = deepcopy(self.model)
        self.mixture_parameters = [torch.zeros_like(p) for p in self.model.parameters()]
        self._sync_counter = 0

    @property
    def required_config_fields(self) -> List[str]:
        return ["mixture_weight", "sync_gap"]

    def communicate(self, target: "APFLServer") -> None:
        self._sync_counter += 1
        message = {
            "client_id": self.client_id,
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        if self._sync_counter == self.config.sync_gap:
            # if the current iteration reaches the synchronization gap,
            # transmit the local model parameters to the server
            # otherwise, only transmit the metrics
            message["parameters"] = self.get_detached_model_parameters()
            # reset the synchronization counter
            self._sync_counter = 0
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        if self._sync_counter == 0:
            # just received the global model parameters
            assert "parameters" in self._received_messages, "No global model parameters received."
            # update the local model parameters
            self.set_parameters(self._received_messages["parameters"])
        # update the mixture parameters
        # which is the convex combination of the local model parameters
        # and the personalized model parameters
        for p, p_per, p_mixture in zip(
            self.model.parameters(),
            self.model_per.parameters(),
            self.mixture_parameters,
        ):
            p_mixture.data = self.config.mixture_weight * p.data + (1 - self.config.mixture_weight) * p_per.data
        # NOTE: we use the optimizer to update the local model parameters automatically in `self.train`,
        # but we need to update the personalized model parameters manually in `self.train_per`,
        # since the gradients are **NOT** computed at the personalized model parameters
        # but at the mixture parameters
        self.solve_inner()  # alias of `train`
        self.train_per()
        self.lr_scheduler.step()

    def train(self) -> None:
        """Train (the copy of) the global model and with local data."""
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
                    # self.optimizer.step(local_weights=self._cached_parameters)
                    self.optimizer.step()
                    # free memory
                    del X, y, output, loss

    def train_per(self) -> None:
        """Train the personalized model with local data."""
        self.model_per.train()
        with tqdm(
            range(self.config.num_epochs),
            total=self.config.num_epochs,
            mininterval=1.0,
            disable=self.config.verbose < 2,
            leave=False,
        ) as pbar:
            for epoch in pbar:
                # compute gradients at the mixture parameters
                grads = self.compute_gradients(at=self.mixture_parameters)
                # update the personalized model parameters via (stochastic) gradient descent
                for p, g in zip(self.model_per.parameters(), grads):
                    p.data = p.data.add(g, alpha=-self.lr_scheduler.get_last_lr()[0])
                # free memory
                del grads

    @torch.no_grad()
    def evaluate(self, part: str) -> Dict[str, float]:
        """Evaluate the model and personalized model
        on the given part of the dataset.

        Parameters
        ----------
        part : str
            The part of the dataset to evaluate on,
            can be either "train" or "val".

        Returns
        -------
        Dict[str, float]
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
                        sum([m[k] * m["num_samples"] for m in _metrics]) / self._metrics[part]["num_samples"]
                    )
            # compute gradient norm of the models
            self._metrics[part][f"grad_norm{suffix}"] = self.get_gradients(norm="fro")
            # free memory
            del X, y, logits
        return self._metrics[part]
