"""
"""

import random
import warnings
from abc import ABC, abstractmethod
from itertools import repeat
from copy import deepcopy
from collections import defaultdict
from typing import Any, Optional, Iterable, List, Tuple, Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from bib_lookup import CitationMixin
from easydict import EasyDict as ED
from tqdm.auto import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch_ecg.utils import ReprMixin, add_docstring, get_kwargs

from .utils.loggers import LoggerManager
from .data_processing.fed_dataset import FedDataset
from .optimizers import get_optimizer


__all__ = [
    "Server",
    "Client",
    "ServerConfig",
    "ClientConfig",
    "ClientMessage",
]


class ServerConfig(ReprMixin):
    """Configs for the Server.

    Parameters
    ----------
    algorithm : str
        The algorithm name.
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    txt_logger : bool, default True
        Whether to use txt logger.
    csv_logger : bool, default True
        Whether to use csv logger.
    json_logger : bool, default True
        Whether to use json logger.
    eval_every : int, default 1
        The number of iterations to evaluate the model.
    verbose : int, default 1
        The verbosity level.
    **kwargs : dict, optional
        The other arguments,
        will be set as attributes of the class.

    """

    __name__ = "ServerConfig"

    def __init__(
        self,
        algorithm: str,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        txt_logger: bool = True,
        csv_logger: bool = False,
        json_logger: bool = True,
        eval_every: int = 1,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        self.algorithm = algorithm
        self.num_iters = num_iters
        self.num_clients = num_clients
        self.clients_sample_ratio = clients_sample_ratio
        self.txt_logger = txt_logger
        self.csv_logger = csv_logger
        if self.csv_logger:
            warnings.warn(
                "CSV logger is not recommended to use, "
                "please use JSON logger instead.",
                RuntimeWarning,
            )
        self.json_logger = json_logger
        self.eval_every = eval_every
        self.verbose = verbose
        for k, v in kwargs.items():
            setattr(self, k, v)

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + list(self.__dict__)


class ClientConfig(ReprMixin):
    """Configs for the Client.

    Parameters
    ----------
    algorithm : str
        The algorithm name.
    optimizer : str
        The name of the optimizer to solve the local (inner) problem.
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float
        The learning rate.
    verbose : int, default 1
        The verbosity level.
    **kwargs : dict, optional
        The other arguments,
        will be set as attributes of the class.

    """

    __name__ = "ClientConfig"

    def __init__(
        self,
        algorithm: str,
        optimizer: str,
        batch_size: int,
        num_epochs: int,
        lr: float,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.verbose = verbose
        for k, v in kwargs.items():
            setattr(self, k, v)

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + list(self.__dict__)


class Node(ReprMixin, ABC):
    """An abstract base class for the server and client nodes."""

    __name__ = "Node"

    @abstractmethod
    def communicate(self, target: "Node") -> None:
        """Communicate with the target node.

        The current node communicates model parameters, gradients, etc. to `target` node.
        For example, a client node communicates its local model parameters to server node via

        .. code-block:: python

            target._received_messages.append(
                ClientMessage(
                    {
                        "client_id": self.client_id,
                        "parameters": self.get_detached_model_parameters(),
                        "train_samples": self.config.num_epochs * self.config.num_steps * self.config.batch_size,
                        "metrics": self._metrics,
                    }
                )
            )

        For a server node, global model parameters are communicated to clients via

        .. code-block:: python

            target._received_messages = {"parameters": self.get_detached_model_parameters()}

        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """Update model parameters, gradients, etc.
        according to `self._reveived_messages`.
        """
        raise NotImplementedError

    def _post_init(self) -> None:
        """Check if all required field in the config are set."""
        assert all(
            [hasattr(self.config, k) for k in self.required_config_fields]
        ), f"missing required config fields: {list(set(self.required_config_fields) - set(self.config.__dict__))}"

    @property
    @abstractmethod
    def required_config_fields(self) -> List[str]:
        """The list of required fields in the config."""
        raise NotImplementedError

    def get_detached_model_parameters(self) -> List[Tensor]:
        """Get the detached model parameters."""
        return [p.detach().clone() for p in self.model.parameters()]

    @staticmethod
    def aggregate_results_from_csv_log(
        df: pd.DataFrame, part: str = "val", metric: str = "acc"
    ) -> np.ndarray:
        """Aggregate the federated results from csv log.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe of the csv log.
        part : str, default "train"
            The part of the log to aggregate.
        metric : str, default "acc"
            The metric to aggregate.

        Returns
        -------
        np.ndarray
            The aggregated results (curve).

        """
        df_part = df[df["part"] == part]
        client_ids = np.unique(
            [
                int(c.split("-")[0].replace("Client", ""))
                for c in df.columns
                if c.startswith("Client")
            ]
        )
        metric_curve = []
        epochs = list(sorted(df_part["epoch"].unique()))
        for epoch in tqdm(epochs, mininterval=1, desc="Aggregating results"):
            df_epoch = df_part[df_part["epoch"] == epoch]
            num_samples = 0
            current_metric = 0
            for client_id in client_ids:
                cols = [f"Client{client_id}-{metric}", f"Client{client_id}-num_samples"]
                df_current = df_epoch[cols].dropna()
                current_metric += (
                    df_current[f"Client{client_id}-{metric}"].values[0]
                    * df_current[f"Client{client_id}-num_samples"].values[0]
                )
                num_samples += df_current[f"Client{client_id}-num_samples"].values[0]
            metric_curve.append(current_metric / num_samples)
        return np.array(metric_curve)

    @staticmethod
    def aggregate_results_from_json_log(
        d: dict, part: str = "val", metric: str = "acc"
    ) -> np.ndarray:
        """Aggregate the federated results from csv log.

        Parameters
        ----------
        d : dict
            The dict of the json/yaml log.
        part : str, default "train"
            The part of the log to aggregate.
        metric : str, default "acc"
            The metric to aggregate.

        Returns
        -------
        np.ndarray
            The aggregated results (curve).

        NOTE that the `d` should be a dict of the following structure:

        .. code-block:: json

            {
                "train": {
                    "client0": [
                        {
                            "epoch": 1,
                            "step": 1,
                            "time": "2020-01-01 00:00:00",
                            "loss": 0.1,
                            "acc": 0.2,
                            "top3_acc": 0.3,
                            "top5_acc": 0.4,
                            "num_samples": 100
                        }
                    ]
                },
                "val": {
                    "client0": [
                        {
                            "epoch": 1,
                            "step": 1,
                            "time": "2020-01-01 00:00:00",
                            "loss": 0.1,
                            "acc": 0.2,
                            "top3_acc": 0.3,
                            "top5_acc": 0.4,
                            "num_samples": 100
                        }
                    ]
                }
            }

        """
        epochs = list(
            sorted(np.unique([item["epoch"] for _, v in d[part].items() for item in v]))
        )
        metric_curve = [[] for _ in range(len(epochs))]
        num_samples = [0 for _ in range(len(epochs))]
        for _, v in tqdm(
            d[part].items(),
            mininterval=1,
            desc="Aggregating results",
            total=len(d[part]),
            unit="client",
        ):
            for item in v:
                idx = epochs.index(item["epoch"])
                metric_curve[idx].append(item[metric] * item["num_samples"])
                num_samples[idx] += item["num_samples"]
        return np.array([sum(v) / num_samples[i] for i, v in enumerate(metric_curve)])


class Server(Node, CitationMixin):
    """The class to simulate the server node.

    The server node is responsible for communicating with clients,
    and perform the aggregation of the local model parameters (and/or gradients),
    and update the global model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained (optimized).
    dataset : FedDataset
        The dataset to be used for training.
    config : ServerConfig
        The configs for the server.
    client_config : ClientConfig
        The configs for the clients.
    lazy : bool, default False
        Whether to use lazy initialization
        for the client nodes.

    """

    __name__ = "Server"

    def __init__(
        self,
        model: nn.Module,
        dataset: FedDataset,
        config: ServerConfig,
        client_config: ClientConfig,
        lazy: bool = False,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.criterion = deepcopy(dataset.criterion)
        assert isinstance(config, self.config_cls["server"]), (
            f"(server) config should be an instance of "
            f"{self.config_cls['server']}, but got {type(config)}."
        )
        self.config = config
        if not hasattr(self.config, "verbose"):
            self.config.verbose = get_kwargs(ServerConfig)["verbose"]
            warnings.warn(
                "The `verbose` attribute is not found in the config, "
                f"set it to the default value {self.config.verbose}.",
                RuntimeWarning,
            )
        self.device = torch.device("cpu")
        assert isinstance(client_config, self.config_cls["client"]), (
            f"client_config should be an instance of "
            f"{self.config_cls['client']}, but got {type(client_config)}."
        )
        self._client_config = client_config
        if not hasattr(self._client_config, "verbose"):
            # self._client_config.verbose = get_kwargs(ClientConfig)["verbose"]
            self._client_config.verbose = self.config.verbose  # set to server's verbose
            warnings.warn(
                "The `verbose` attribute is not found in the client_config, "
                f"set it to the default value {self._client_config.verbose}.",
                RuntimeWarning,
            )

        logger_config = dict(
            txt_logger=self.config.txt_logger,
            csv_logger=self.config.csv_logger,
            json_logger=self.config.json_logger,
            algorithm=self.config.algorithm,
            model=self.model.__class__.__name__,
            dataset=dataset.__class__.__name__,
            verbose=self.config.verbose,
        )
        self._logger_manager = LoggerManager.from_config(logger_config)

        # set batch_size, in case of centralized training
        setattr(self.config, "batch_size", client_config.batch_size)

        self._received_messages = []
        self._num_communications = 0

        self.n_iter = 0

        if not lazy:
            self._clients = self._setup_clients(dataset, client_config)

        self._post_init()

    def _setup_clients(
        self, dataset: FedDataset, client_config: ClientConfig
    ) -> List[Node]:
        """Setup the clients.

        Parameters
        ----------
        dataset : FedDataset
            The dataset to be used for training the local models.
        client_config : ClientConfig
            The configs for the clients.

        Returns
        -------
        List[Node]
            A list of clients.

        """
        print("setup clients...")
        return [
            self.client_cls(
                client_id, device, deepcopy(self.model), dataset, client_config
            )
            for client_id, device in zip(
                range(self.config.num_clients), self._allocate_devices()
            )
        ]

    def _allocate_devices(self) -> List[torch.device]:
        """Allocate devices for clients, can be used in :meth:`_setup_clients`."""
        print("allocate devices...")
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return list(repeat(torch.device("cpu"), self.config.num_clients))
        return [
            torch.device(f"cuda:{i%num_gpus}") for i in range(self.config.num_clients)
        ]

    def _sample_clients(self) -> List[int]:
        """Sample clients for each iteration."""
        k = int(self.config.num_clients * self.config.clients_sample_ratio)
        return random.sample(range(self.config.num_clients), k)

    def _communicate(self, target: "Client") -> None:
        """Broadcast to target client, and maintain state variables."""
        self.communicate(target)
        self._num_communications += 1

    def _update(self) -> None:
        """Server update, and clear cached messages from clients of the previous iteration."""
        self._logger_manager.log_message("Server update...")
        if len(self._received_messages) == 0:
            warnings.warn(
                "No message received from the clients, unable to update server model"
            )
            return
        assert all(
            [isinstance(m, ClientMessage) for m in self._received_messages]
        ), "received messages must be of type `ClientMessage`"
        self.update()
        self._received_messages = (
            []
        )  # clear messages received in the previous iteration
        self._logger_manager.log_message("Server update finished...")

    def train(
        self, mode: str = "federated", extra_configs: Optional[dict] = None
    ) -> None:
        """The main training loop.

        Parameters
        ----------
        mode : {"federated", "centralized"}, optional
            The mode of training, by default "federated", case-insensitive.
        extra_configs : dict, optional
            The extra configs for the training `mode`.

        Returns
        -------
        None

        """
        if mode.lower() == "federated":
            self.train_federated(extra_configs)
        elif mode.lower() == "centralized":
            self.train_centralized(extra_configs)
        else:
            raise ValueError(f"mode {mode} is not supported")

    def train_centralized(self, extra_configs: Optional[dict] = None) -> None:
        """Centralized training, conducted only on the server node.

        This is used as a baseline for comparison.

        Parameters
        ----------
        extra_configs : dict, optional
            The extra configs for centralized training.

        Returns
        -------
        None

        """
        self._logger_manager.log_message("Training centralized...")
        extra_configs = ED(extra_configs or {})

        batch_size = extra_configs.get("batch_size", self.config.batch_size)
        train_loader, val_loader = self.dataset.get_dataloader(
            batch_size, batch_size, None
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        self.model.to(device)

        criterion = deepcopy(self.dataset.criterion)
        lr = extra_configs.get("lr", 1e-2)
        optimizer = extra_configs.get("optimizer", SGD(self.model.parameters(), lr))
        scheduler = extra_configs.get(
            "scheduler", LambdaLR(optimizer, lambda epoch: 1 / (0.01 * epoch + 1))
        )

        epoch_losses = []
        self.n_iter, global_step = 0, 0
        for self.n_iter in range(self.config.num_iters):
            with tqdm(
                total=len(train_loader.dataset),
                desc=f"Epoch {self.n_iter+1}/{self.config.num_iters}",
                unit="sample",
                mininterval=1.0,
            ) as pbar:
                epoch_loss = []
                batch_losses = []
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    batch_losses.append(loss.item())
                    optimizer.step()
                    global_step += 1
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    pbar.update(data.shape[0])
                epoch_loss.append(sum(batch_losses) / len(batch_losses))
                if (self.n_iter + 1) % self.config.eval_every == 0:
                    print("evaluating...")
                    metrics = self.evaluate_centralized(val_loader)
                    self._logger_manager.log_metrics(
                        None,
                        metrics,
                        step=global_step,
                        epoch=self.n_iter + 1,
                        part="val",
                    )
                    metrics = self.evaluate_centralized(train_loader)
                    self._logger_manager.log_metrics(
                        None,
                        metrics,
                        step=global_step,
                        epoch=self.n_iter + 1,
                        part="train",
                    )
                scheduler.step()

        self.model.to(self.device)  # move to the original device
        self._logger_manager.log_message("Centralized training finished...")
        self._logger_manager.flush()
        self._logger_manager.reset()

    def train_federated(self, extra_configs: Optional[dict] = None) -> None:
        """Federated (distributed) training,
        conducted on the clients and the server.

        Parameters
        ----------
        extra_configs : dict, optional
            The extra configs for federated training.

        Returns
        -------
        None

        TODO
        ----
        Run clients training in parallel.

        """
        self._logger_manager.log_message("Training federated...")
        self.n_iter = 0
        for self.n_iter in range(self.config.num_iters):
            selected_clients = self._sample_clients()
            with tqdm(
                total=len(selected_clients),
                desc=f"Iter {self.n_iter+1}/{self.config.num_iters}",
                unit="client",
                mininterval=1.0,
            ) as pbar:
                for client_id in selected_clients:
                    client = self._clients[client_id]
                    self._communicate(client)
                    if (
                        self.n_iter > 0
                        and (self.n_iter + 1) % self.config.eval_every == 0
                    ):
                        for part in self.dataset.data_parts:
                            metrics = client.evaluate(part)
                            # print(f"metrics: {metrics}")
                            self._logger_manager.log_metrics(
                                client_id,
                                metrics,
                                step=self.n_iter,
                                epoch=self.n_iter,
                                part=part,
                            )
                    client._update()
                    client._communicate(self)
                    pbar.update(1)
                if self.n_iter > 0 and (self.n_iter + 1) % self.config.eval_every == 0:
                    # TODO: fix potential errors in the function below
                    self.aggregate_client_metrics()
                self._update()
        self._logger_manager.log_message("Federated training finished...")
        self._logger_manager.flush()
        self._logger_manager.reset()

    def evaluate_centralized(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given dataloader on the server node.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader for evaluation.

        Returns
        -------
        metrics : dict
            The metrics of the model on the given dataloader.

        """
        metrics = []
        for (X, y) in dataloader:
            X, y = X.to(self.model.device), y.to(self.model.device)
            probs = self.model(X)
            metrics.append(self.dataset.evaluate(probs, y))
        num_samples = sum([m["num_samples"] for m in metrics])
        metrics_names = [k for k in metrics[0] if k != "num_samples"]
        metrics = {
            k: sum([m[k] * m["num_samples"] for m in metrics]) / num_samples
            for k in metrics_names
        }
        metrics["num_samples"] = num_samples
        return metrics

    def aggregate_client_metrics(self) -> None:
        """Aggregate the metrics transmitted from the clients."""
        if not any(["metrics" in m for m in self._received_messages]):
            raise ValueError("no metrics received from clients")
        for part in self.dataset.data_parts:
            metrics = defaultdict(float)
            for m in self._received_messages:
                if "metrics" not in m:
                    continue
                for k, v in m["metrics"][part].items():
                    if k != "num_samples":
                        metrics[k] += (
                            m["metrics"][part][k] * m["metrics"][part]["num_samples"]
                        )
                    else:
                        metrics[k] += m["metrics"][part][k]
            for k in metrics:
                if k != "num_samples":
                    metrics[k] /= metrics["num_samples"]
            self._logger_manager.log_metrics(
                None,
                dict(metrics),
                step=self.n_iter + 1,
                epoch=self.n_iter + 1,
                part=part,
            )

    def add_parameters(self, params: Iterable[Parameter], ratio: float) -> None:
        """Update the server's parameters with the given parameters.

        Parameters
        ----------
        params : Iterable[torch.nn.Parameter]
            The parameters to be added.
        ratio : float
            The ratio of the parameters to be added.

        Returns
        -------
        None

        """
        for server_param, param in zip(self.model.parameters(), params):
            server_param.data.add_(
                param.data.detach().clone().to(self.device), alpha=ratio
            )

    def avg_parameters(self, size_aware: bool = False, inertia: float = 0.0) -> None:
        """Update the server's parameters via
        averaging the parameters received from the clients.

        Parameters
        ----------
        size_aware : bool, default False
            Whether to use the size-aware averaging.
        inertia : float, default 0.0
            The weight of the previous parameters.

        Returns
        -------
        None

        """
        assert 0.0 <= inertia < 1.0, "`inertia` should be in [0, 1)"
        if len(self._received_messages) == 0:
            return
        for param in self.model.parameters():
            param.data.mul_(inertia)
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for m in self._received_messages:
            ratio = (
                m["train_samples"] / total_samples
                if size_aware
                else 1 / len(self._received_messages)
            ) * (1 - inertia)
            self.add_parameters(m["parameters"], ratio)

    def update_gradients(self) -> None:
        """Update the server's gradients."""
        if len(self._received_messages) == 0:
            return
        assert all(
            ["gradients" in m for m in self._received_messages]
        ), "some clients have not sent gradients yet"
        # self.model.zero_grad()
        for mp, gd in zip(
            self.model.parameters(), self._received_messages[0]["gradients"]
        ):
            mp.grad = torch.zeros_like(gd).to(self.device)
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for rm in self._received_messages:
            for mp, gd in zip(self.model.parameters(), rm["gradients"]):
                mp.grad.add_(
                    gd.detach().clone().to(self.device),
                    alpha=rm["train_samples"] / total_samples,
                )

    def get_client_data(self, client_idx: int) -> Tuple[Tensor, Tensor]:
        """Get all the data of the given client.

        This method is a helper function for fast access
        to the data of the given client.

        Parameters
        ----------
        client_idx : int
            The index of the client.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Input data and labels of the given client.

        """
        if client_idx >= len(self._clients):
            raise ValueError(f"client_idx should be less than {len(self._clients)}")

        return self._clients[client_idx].get_all_data()

    def get_client_model(self, client_idx: int) -> torch.nn.Module:
        """Get the model of the given client.

        This method is a helper function for fast access
        to the model of the given client.

        Parameters
        ----------
        client_idx : int
            The index of the client.

        Returns
        -------
        torch.nn.Module
            The model of the given client.

        """
        if client_idx >= len(self._clients):
            raise ValueError(f"client_idx should be less than {len(self._clients)}")

        return self._clients[client_idx].model

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "config",
            "client_config",
        ]

    @property
    @abstractmethod
    def client_cls(self) -> type:
        """Class of the client node."""
        raise NotImplementedError

    @property
    @abstractmethod
    def config_cls(self) -> Dict[str, type]:
        """Class of the client node config and server node config.

        Keys are "client" and "server".
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def doi(self) -> Union[str, List[str]]:
        raise NotImplementedError


class Client(Node):
    """The class to simulate the client node.

    The client node is responsible for training the local models,
    and communicating with the server node.

    Parameters
    ----------
    client_id : int
        The id of the client.
    device : torch.device
        The device to train the model on.
    model : torch.nn.Module
        The model to train.
    dataset : FedDataset
        The dataset to train on.
    config : ClientConfig
        The config for the client.

    """

    __name__ = "Client"

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        model: nn.Module,
        dataset: FedDataset,
        config: ClientConfig,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.criterion = deepcopy(dataset.criterion)
        self.dataset = dataset
        self.config = config

        self.optimizer = get_optimizer(
            optimizer_name=config.optimizer,
            params=self.model.parameters(),
            config=config,
        )
        self.train_loader, self.val_loader = self.dataset.get_dataloader(
            self.config.batch_size, self.config.batch_size, self.client_id
        )

        self._cached_parameters = None
        self._received_messages = {}
        self._metrics = {}

        self._post_init()

    def _communicate(self, target: "Server") -> None:
        """Check validity and send messages to the server,
        and maintain state variables.

        Parameters
        ----------
        target : Server
            The server to communicate with.

        Returns
        -------
        None

        """
        # check validity of self._metrics
        for part, metrics in self._metrics.items():
            assert isinstance(metrics, dict), (
                f"metrics for {part} should be a dict, "
                f"but got {type(metrics).__name__}"
            )
            assert "num_samples" in metrics, (
                "In order to let the server aggregate the metrics, "
                f"metrics for {part} should have key `num_samples`, "
                f"but got {metrics.keys()}"
            )
        self.communicate(target)
        target._num_communications += 1
        self._metrics = {}  # clear the metrics

    def _update(self) -> None:
        """Client update, and clear cached messages
        from the server of the previous iteration.
        """
        self.update()
        self._received_messages = {}  # clear the received messages

    @abstractmethod
    def train(self) -> None:
        """Main part of inner loop solver.

        Basic example:

        .. code-block:: python

            self.model.train()
            epoch_losses = []
            for epoch in range(self.config.num_epochs):
                batch_losses = []
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    batch_losses.append(loss.item())
                epoch_losses.append(sum(batch_losses) / len(batch_losses))

        """
        raise NotImplementedError

    @add_docstring(train.__doc__)
    def solve_inner(self) -> None:
        """alias of `train`"""
        self.train()

    def sample_data(self) -> Tuple[Tensor, Tensor]:
        """Sample data for training."""
        return next(iter(self.train_loader))

    @torch.no_grad()
    def evaluate(self, part: str) -> Dict[str, float]:
        """Evaluate the model on the given part of the dataset.

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
        self.model.eval()
        # _metrics = []
        data_loader = self.val_loader if part == "val" else self.train_loader
        if data_loader is None:
            self._metrics[part] = {"num_samples": 0}
            return self._metrics[part]
        all_logits, all_labels = [], []
        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            all_logits.append(logits)
            all_labels.append(y)
            # _metrics.append(self.dataset.evaluate(logits, y))
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        self._metrics[part] = {"num_samples": len(all_labels)}
        self._metrics[part].update(self.dataset.evaluate(all_logits, all_labels))
        return self._metrics[part]

    def set_parameters(self, params: Iterable[Parameter]) -> None:
        """Set the parameters of the (local) model on the client.

        Parameters
        ----------
        params : Iterable[torch.nn.Parameter]
            The parameters to set.

        Returns
        -------
        None

        """
        for client_param, param in zip(self.model.parameters(), params):
            client_param.data = param.data.detach().clone().to(self.device)

    def get_gradients(self) -> List[Tensor]:
        """Get the gradients of the (local) model on the client."""
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def get_all_data(self) -> Tuple[Tensor, Tensor]:
        """Get all the data on the client.

        This method is a helper function for fast access
        to the data on the client,
        including both training and validation data;
        both features and labels.
        """
        feature_train, label_train = self.train_loader.dataset[:]
        if self.val_loader is None:
            return feature_train, label_train
        feature_val, label_val = self.val_loader.dataset[:]
        feature = torch.cat([feature_train, feature_val], dim=0)
        label = torch.cat([label_train, label_val], dim=0)
        return feature, label

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "client_id",
            "config",
        ]


class ClientMessage(dict):
    """A class used to specify required fields
    for a message from client to server.

    Parameters
    ----------
    client_id : int
        The id of the client.
    train_samples : int
        The number of samples used for training on the client.
    metrics : dict
        The metrics evaluated on the client.
    **kwargs : dict, optional
        Extra message to be sent to the server.

    """

    __name__ = "ClientMessage"

    def __init__(
        self, client_id: int, train_samples: int, metrics: dict, **kwargs
    ) -> None:
        super().__init__(
            client_id=client_id, train_samples=train_samples, metrics=metrics, **kwargs
        )
