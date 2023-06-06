"""
"""

import warnings
from copy import deepcopy
from typing import List, Optional, Dict, Any

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from ...optimizers import get_optimizer
from ...utils.misc import get_scheduler


__all__ = [
    "DittoServerConfig",
    "DittoClientConfig",
    "DittoServer",
    "DittoClient",
]


class DittoServerConfig(ServerConfig):
    """Server config for the Ditto algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``log_dir`` : str or Path, optional
            The log directory.
            If not specified, will use the default log directory.
            If not absolute, will be relative to the default log directory.
        - ``txt_logger`` : bool, default True
            Whether to use txt logger.
        - ``csv_logger`` : bool, default False
            Whether to use csv logger.
        - ``json_logger`` : bool, default True
            Whether to use json logger.
        - ``eval_every`` : int, default 1
            The number of iterations to evaluate the model.
        - ``visiable_gpus`` : Sequence[int], optional
            Visable GPU IDs for allocating devices for clients.
            Defaults to use all GPUs if available.
        - ``seed`` : int, default 0
            The random seed.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``gpu_proportion`` : float, default 0.2
            The proportion of clients to use GPU.
            Used to similate the system heterogeneity of the clients.
            Not used in the current version, reserved for future use.

    """

    __name__ = "DittoServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "Ditto",
            num_iters,
            num_clients,
            clients_sample_ratio,
            **kwargs,
        )


class DittoClientConfig(ClientConfig):
    """Client config for the Ditto algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    optimizer : str, default "ProxSGD"
        The name of the optimizer to solve the local (inner) problem.
    optimizer_per : str, default "SGD"
        The name of the optimizer to solve the personalization problem.
    prox : float, default 0.01
        Coefficient of the proximal term.
    lr : float, default 1e-2
        The learning rate.
    lr_per : float, optional
        The learning rate for personalization.
        If not specified, ``lr_per`` will be set to ``lr``.
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

    __name__ = "DittoClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        optimizer: str = "ProxSGD",
        optimizer_per: str = "SGD",
        prox: float = 0.01,
        lr: float = 1e-2,
        lr_per: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn("The `algorithm` argument fixed to `Ditto`.", RuntimeWarning)
        super().__init__(
            "Ditto",
            optimizer,
            batch_size,
            num_epochs,
            lr,
            prox=prox,
            optimizer_per=optimizer_per,
            lr_per=lr_per if lr_per is not None else lr,
            **kwargs,
        )


@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.", "Server node for the Ditto algorithm."
    )
    .replace("ServerConfig", "DittoServerConfig")
    .replace("ClientConfig", "DittoClientConfig")
)
class DittoServer(Server):
    """Server node for the Ditto algorithm."""

    __name__ = "DittoServer"

    @property
    def client_cls(self) -> type:
        return DittoClient

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "DittoClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    @add_docstring(Server.update)
    def update(self) -> None:
        # sum of received parameters, with self.model.parameters() as its container
        self.avg_parameters()

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": DittoServerConfig,
            "client": DittoClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2012.04221"]


@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.", "Client node for the Ditto algorithm."
    ).replace("ClientConfig", "DittoClientConfig")
)
class DittoClient(Client):
    """Client node for the Ditto algorithm."""

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
        scheduler_config = {
            k: v for k, v in self.config.scheduler.items() if k != "name"
        }
        self.scheduler_per = get_scheduler(
            scheduler_name=self.config.scheduler["name"],
            optimizer=self.optimizer_per,
            config=scheduler_config,
        )

    @property
    def required_config_fields(self) -> List[str]:
        return ["optimizer_per", "lr_per"]

    def communicate(self, target: "DittoServer") -> None:
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
                "No parameters received from server. "
                "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        self.set_parameters(self._cached_parameters)
        self.solve_inner()  # alias of self.train()
        self.train_per()

    def train(self) -> None:
        """Train (the copy of) the global model with local data."""
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
                    self.optimizer.step(local_weights=self._cached_parameters)
                    # free memory
                    del X, y, output, loss
        self.lr_scheduler.step()

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
            for epoch in pbar:  # local update
                self.model_per.train()
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer_per.zero_grad()
                    output = self.model_per(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer_per.step(local_weights=self._cached_parameters)
                    # free memory
                    del X, y, output, loss
        self.scheduler_per.step()

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
                        sum([m[k] * m["num_samples"] for m in _metrics])
                        / self._metrics[part]["num_samples"]
                    )
            # compute gradient norm of the models
            self._metrics[part][f"grad_norm{suffix}"] = self.get_gradients(norm="fro")
            # free memory
            del X, y, logits
        return self._metrics[part]
