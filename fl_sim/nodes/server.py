"""
The base Server class.
======================

"""

import json
import warnings
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from bib_lookup import CitationMixin
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch_ecg.cfg import CFG
from tqdm.auto import tqdm

from ..data_processing.fed_dataset import FedDataset
from ..messages import ClientMessage, ClientToServer, ServerToClient
from ..models import reset_parameters
from ..utils.loggers import LoggerManager
from ..utils.misc import default_dict_to_dict
from .client import Client
from .configs import ClientConfig, ServerConfig
from .node import Node

__all__ = [
    "Server",
]


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
        for the client nodes. This is useful
        when one wants to do centralized training
        for verification.

    TODO
    ----
    1. Run clients training in parallel.
    2. Use the attribute `_is_convergent` to control the termination of the training.
       This perhaps can be achieved by comparing part of the items in `self._cached_models`

    """

    __name__ = "Server"

    def __init__(
        self,
        model: nn.Module,
        dataset: FedDataset,
        config: ServerConfig,
        client_config: ClientConfig,
        lazy: bool = False,
        is_subprocess: bool = False,
    ) -> None:
        self.model = model
        self.dataset = dataset
        # NOTE: get server val loader from dataset for centralized evaluation
        # TODO: reimplement dataset.get_dataloader, add parameter to get server val loader alone
        _, self.server_val_loader = dataset.get_dataloader()
        self.criterion = deepcopy(dataset.criterion)
        assert isinstance(config, self.config_cls["server"]), (
            f"(server) config should be an instance of " f"{self.config_cls['server']}, but got {type(config)}."
        )
        self.config = config
        if self.config.num_clients is None:
            self.config.num_clients = self.dataset.DEFAULT_TRAIN_CLIENTS_NUM
        self.device = torch.device("cpu")
        assert isinstance(client_config, self.config_cls["client"]), (
            f"client_config should be an instance of " f"{self.config_cls['client']}, but got {type(client_config)}."
        )
        self._client_config = client_config
        # multi-processing
        self._subprocess = is_subprocess
        if self._subprocess:
            # import necessary modules for multi-processing
            from ..utils.multiprocessing import report_progress

            self._report_progress = report_progress

        logger_config = dict(
            log_dir=self.config.log_dir,
            log_suffix=self.config.tag,
            txt_logger=self.config.txt_logger,
            json_logger=self.config.json_logger,
            algorithm=self.config.algorithm,
            model=self.model.__class__.__name__,
            dataset=self.dataset.__class__.__name__,
            verbose=self.config.verbose,
            silent=self._subprocess,
        )
        self._logger_manager = LoggerManager.from_config(logger_config)

        # set batch_size, in case of centralized training
        setattr(self.config, "batch_size", client_config.batch_size)

        # persist the (machine-readable) configs for reproducibility
        configs_json = {
            "server": self.config.model_dump(mode="json"),
            "client": self._client_config.model_dump(mode="json"),
        }
        config_json_path = Path(self._logger_manager.log_dir) / "config.json"
        if config_json_path.exists():  # e.g. re-run in the same directory
            config_json_path.rename(config_json_path.with_suffix(".json.bak"))
        config_json_path.write_text(json.dumps(configs_json, indent=4, ensure_ascii=False))

        # echo the configs to stdout and txt log file
        self._logger_manager.log_message(f"Server Configs:\n{str(self.config)}")
        self._logger_manager.log_message(f"Client Configs:\n{str(self._client_config)}")

        self._received_messages = []
        self._num_communications = 0
        self.n_iter = 0
        self._metrics = {}  # aggregated metrics
        self._cached_metrics = []  # container for caching aggregated metrics
        self._is_convergent = False  # status variable for convergence
        # flag for completing the experiment
        # which will be checked before calling
        # `self.train`, `self.train_centralized`, `self.train_federated`
        self._complete_experiment = False

        self._clients = None
        if not lazy:
            self._setup_clients(self.dataset, self._client_config)

        self._post_init()

        # checks that the client has all the required attributes in config.extra_observes
        for attr in self.config.extra_observes:
            assert hasattr(self, attr), f"{self.__name__} should have attribute {attr} for extra observes."

    def _setup_clients(
        self,
        dataset: Optional[FedDataset] = None,
        client_config: Optional[ClientConfig] = None,
        force: bool = False,
    ) -> None:
        """Setup the clients.

        Parameters
        ----------
        dataset : FedDataset, optional
            The dataset to be used for training the local models,
            defaults to `self.dataset`.
        client_config : ClientConfig, optional
            The configs for the clients,
            defaults to `self._client_config`.
        force : bool, default False
            Whether to force setup the clients.
            If set to True, the clients will be setup
            even if they have been setup before.

        Returns
        -------
        None

        """
        if self._clients is not None and not force:
            return
        elif force:
            # reset the clients
            del self._clients
        self._logger_manager.log_message("Setup clients...")
        dataset = dataset or self.dataset
        client_config = client_config or self._client_config
        self._clients = [
            self.client_cls(client_id, device, deepcopy(self.model), dataset, client_config)
            for client_id, device in tqdm(
                zip(range(self.config.num_clients), self._allocate_devices()),
                desc="Allocating devices",
                total=self.config.num_clients,
                unit="client",
                mininterval=1,
                leave=False,
            )
        ]

    def _allocate_devices(self) -> List[torch.device]:
        """Allocate devices for clients, can be used in :meth:`_setup_clients`."""
        self._logger_manager.log_message("Allocate devices...")
        # num_gpus = torch.cuda.device_count()
        num_gpus = len(self.config.visiable_gpus)
        if num_gpus == 0:
            return list(repeat(torch.device("cpu"), self.config.num_clients))
        return [torch.device(f"cuda:{self.config.visiable_gpus[i%num_gpus]}") for i in range(self.config.num_clients)]

    def _sample_clients(
        self,
        subset: Optional[Sequence[int]] = None,
        clients_sample_ratio: Optional[float] = None,
        clients_sample_num: Optional[int] = None,
    ) -> List[int]:
        """Sample clients for each iteration.

        Parameters
        ----------
        subset : Sequence[int], optional
            The subset of clients to be sampled from,
            defaults to `None`, which means all clients.
        clients_sample_ratio : float, optional
            The ratio of clients to be sampled,
            defaults to `None`, which means `self.config.clients_sample_ratio`.
        clients_sample_num : int, optional
            The number of clients to be sampled.
            If not `None`, will override `clients_sample_ratio`.

        Returns
        -------
        List[int]
            The list of selected client ids.

        """
        if subset is None:
            subset = range(self.config.num_clients)
        if clients_sample_ratio is None:
            clients_sample_ratio = self.config.clients_sample_ratio
        if clients_sample_num is not None:
            k = min(len(subset), max(1, clients_sample_num))
        else:
            k = min(len(subset), max(1, round(clients_sample_ratio * len(subset))))
        # return random.sample(subset, k=k)
        # random.sample does not support numpy array input
        return np.random.choice(subset, k, replace=False).tolist()

    def communicate(self, target: "Client") -> None:
        """Broadcast the current model parameters to the target client.

        This is the default (typed-API) implementation; subclasses override it
        for algorithm-specific payloads. The typed-API way to send additional
        data is building a :class:`~fl_sim.messages.ServerToClient` and calling
        :meth:`Client.receive` on the target; the legacy way (overriding
        `communicate` and writing `target._received_messages` directly)
        keeps working.

        Parameters
        ----------
        target : Client
            The client to broadcast to.

        Returns
        -------
        None

        """
        target.receive(ServerToClient(parameters=self.get_detached_model_parameters()))

    def receive_one(self, message: ClientToServer) -> None:
        """Receive a typed message uploaded from a client.

        The message is converted to the legacy :class:`ClientMessage` and
        appended to ``self._received_messages``.

        Parameters
        ----------
        message : ClientToServer
            The typed message uploaded from a client.

        Returns
        -------
        None

        """
        self._received_messages.append(ClientMessage.from_typed(message))

    def _communicate(self, target: "Client") -> None:
        """Broadcast to target client, and maintain state variables."""
        self.communicate(target)
        self._num_communications += 1

    def _update(self) -> None:
        """Server update, and clear cached messages from clients of the previous iteration."""
        self._logger_manager.log_message("Server update...")
        if len(self._received_messages) == 0:
            warnings.warn(
                "No message received from the clients, unable to update server model.",
                RuntimeWarning,
            )
            return
        assert all([isinstance(m, ClientMessage) for m in self._received_messages]), (
            "received messages must be of type `ClientMessage`, "
            f"but got {[type(m) for m in self._received_messages if not isinstance(m, ClientMessage)]}"
        )
        self.update()
        # free the memory of the received messages
        del self._received_messages
        self._received_messages = []  # clear messages received in the previous iteration
        self._logger_manager.log_message("Server update finished...")

    def train(self, mode: str = "federated", extra_configs: Optional[dict] = None) -> None:
        """The main training loop.

        Parameters
        ----------
        mode : {"federated", "centralized", "local"}, optional
            The mode of training, by default "federated", case-insensitive.
        extra_configs : dict, optional
            The extra configs for the training `mode`.

        Returns
        -------
        None

        """
        if self._complete_experiment:
            # reset before training if a previous experiment is completed
            self._reset(reset_clients=(mode.lower() != "centralized"))
        self._complete_experiment = False
        if mode.lower() == "federated":
            self.train_federated(extra_configs)
        elif mode.lower() == "centralized":
            self.train_centralized(extra_configs)
        elif mode.lower() == "local":
            self.train_local(extra_configs)
        else:
            raise ValueError(f"mode {mode} is not supported")
        self._complete_experiment = True

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
        if self._complete_experiment:
            # reset before training if a previous experiment is completed
            # no need to reset clients for centralized training
            self._reset(reset_clients=False)

        self._logger_manager.log_message("Training centralized...")
        extra_configs = CFG(extra_configs or {})

        batch_size = extra_configs.get("batch_size", self.config.batch_size)
        train_loader, val_loader = self.dataset.get_dataloader(batch_size, batch_size, None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        self.model.to(device)

        criterion = deepcopy(self.dataset.criterion)
        lr = extra_configs.get("lr", 1e-2)
        optimizer = extra_configs.get("optimizer", SGD(self.model.parameters(), lr))
        scheduler = extra_configs.get("scheduler", LambdaLR(optimizer, lambda epoch: 1 / (0.01 * epoch + 1)))

        self._complete_experiment = False
        epoch_losses = []
        self.n_iter, global_step = 0, 0
        # report progress for multi-processing
        if self._subprocess:
            self._report_progress(
                n_iter=self.n_iter,
                num_iters=self.config.num_iters,
                current_client_progress=None,
                selected_clients_count=None,
                training_phase="training",
            )
        for self.n_iter in range(self.config.num_iters):
            with tqdm(
                total=len(train_loader.dataset),
                desc=f"Epoch {self.n_iter+1}/{self.config.num_iters}",
                unit="sample",
                mininterval=1 if torch.cuda.is_available() else 10,
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
                    if self.config.verbose >= 2:
                        pbar.set_postfix(
                            **{
                                "loss (batch)": loss.item(),
                                "lr": scheduler.get_last_lr()[0],
                            }
                        )
                    pbar.update(data.shape[0])
                    # report progress for multi-processing
                    if self._subprocess:
                        self._report_progress(
                            n_iter=self.n_iter + 1,
                            num_iters=self.config.num_iters,
                            current_client_progress=None,
                            selected_clients_count=None,
                            training_phase="training",
                        )
                    # free memory
                    del data, target, output, loss
                epoch_loss.append(sum(batch_losses) / len(batch_losses))
                if (self.n_iter + 1) % self.config.eval_every == 0:
                    self._logger_manager.log_message("Evaluating...")
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
        # self._logger_manager.reset()

        self._complete_experiment = True

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
        if self._clients is None:
            self._clients = self._setup_clients(self.dataset, self._client_config)

        if self._complete_experiment:
            # reset before training if a previous experiment is completed
            self._reset()

        self._complete_experiment = False
        self._logger_manager.log_message("Training federated...")
        self.n_iter = 0
        # report progress for multi-processing
        if self._subprocess:
            self._report_progress(
                n_iter=self.n_iter,
                num_iters=self.config.num_iters,
                current_client_progress=None,
                selected_clients_count=None,
                training_phase="training",
            )
            # DEBUG message
            # print(f"DEBUG: report progress for multi-processing using {self._report_progress.__name__}")
        with tqdm(
            range(self.config.num_iters),
            total=self.config.num_iters,
            desc=f"{self.config.algorithm} training",
            unit="iter",
            mininterval=1,  # usually useless since a full iteration takes a very long time
        ) as outer_pbar:
            for self.n_iter in outer_pbar:
                selected_clients = self._sample_clients()
                self.selected_clients = selected_clients
                with tqdm(
                    total=len(selected_clients),
                    desc=f"Iter {self.n_iter+1}/{self.config.num_iters}",
                    unit="client",
                    mininterval=max(1, len(selected_clients) // 50),
                    disable=self.config.verbose < 1,
                    leave=False,
                ) as pbar:
                    # count for client training progress for multi-processing
                    if self._subprocess:
                        count_client = 0
                    # NOTE: evaluate centralized model on server val loader
                    if self.server_val_loader:
                        # TODO: reimplement evaluate logic
                        metrics = self.evaluate_centralized(self.server_val_loader)
                        self._logger_manager.log_metrics(
                            None,
                            metrics,
                            step=self._num_communications,  # NOTE: use number of communications as step for now
                            epoch=self.n_iter,
                            part="val",
                        )
                    self.pre_iter()
                    evaluate = self._should_evaluate()
                    for client_id in selected_clients:
                        client = self._clients[client_id]
                        self._broadcast(client)
                        self._local_step(client, evaluate=evaluate)
                        pbar.update(1)
                        # report progress for multi-processing
                        if self._subprocess:
                            count_client += 1
                            self._report_progress(
                                n_iter=self.n_iter + 1,
                                num_iters=self.config.num_iters,
                                current_client_progress=count_client,
                                selected_clients_count=len(selected_clients),
                                training_phase="training",
                            )
                            # DEBUG message
                            # print("DEBUG: report progress for multi-processing")
                    self._aggregate_metrics(evaluate=evaluate)
                    self._global_step()
        self._logger_manager.log_message("Federated training finished...")
        self._logger_manager.flush()

        self._complete_experiment = True

    def _should_evaluate(self) -> bool:
        """Whether to evaluate the (train-part) metrics on the clients in this iteration."""
        return self.n_iter > 0 and (self.n_iter + 1) % self.config.eval_every == 0

    def _broadcast(self, client: "Client") -> None:
        """Downlink phase: broadcast to the client, and let it accept the parameters.

        The broadcast is done by :meth:`Server.communicate` (overridable per algorithm),
        and the client applies the received parameters via
        :meth:`Client.accept_parameters` (also overridable per algorithm).
        """
        # server communicates with client, typically broadcasting the global model to the client
        self._communicate(client)
        # NOTE: before evaluating (and updating), the client should accept parameters;
        # if the key in received_messages is not "parameters", accept_parameters
        # needs to be reimplemented.
        client.accept_parameters()

    def _local_step(self, client: "Client", evaluate: bool = False) -> None:
        """Local phase on one client: evaluate (optional) → update → upload.

        NOTE: evaluation (if any) is always executed BEFORE the client's local
        update, so that the metrics are computed on the model just received
        from the server, rather than on the locally-updated (possibly
        personalized) model. This ordering is guaranteed here by construction.

        Parameters
        ----------
        client : Client
            The client to run the local step on.
        evaluate : bool, optional
            Whether to evaluate (and log) the train-part metrics on the client
            before the local update, by default False.

        """
        if evaluate:
            # only evaluate train metrics on clients
            metrics = client.evaluate("train")
            self._logger_manager.log_metrics(
                client.client_id,
                metrics,
                step=self.n_iter,
                epoch=self.n_iter,
                part="train",
            )
        # client trains the model and perhaps updates other local variables
        client._update()
        # client communicates with server, typically sending the local model
        # and evaluated local metrics to the server,
        # and perhaps other local variables (e.g. gradients, etc.)
        client._communicate(self)

    def _aggregate_metrics(self, evaluate: bool = False) -> None:
        """Aggregate phase: the server aggregates the client metrics of this iteration.

        Parameters
        ----------
        evaluate : bool, optional
            Whether client metrics were evaluated (and logged) this iteration,
            by default False. Aggregation happens only when they were.

        """
        if evaluate:
            self.aggregate_client_metrics()

    def _global_step(self) -> None:
        """Global phase: the server updates the global model (and other global variables)."""
        self._update()

    def train_local(self, extra_configs: Optional[dict] = None) -> None:
        """Local training, conducted on the clients,
        without any communication with the server. Used for comparison.

        Parameters
        ----------
        extra_configs : dict, optional
            The extra configs for local training.

        Returns
        -------
        None

        """
        if self._clients is None:
            self._clients = self._setup_clients(self.dataset, self._client_config)

        if self._complete_experiment:
            # reset before training if a previous experiment is completed
            self._reset()

        self._complete_experiment = False
        self._logger_manager.log_message("Training local...")
        self.n_iter = 0
        # report progress for multi-processing
        if self._subprocess:
            self._report_progress(
                n_iter=self.n_iter,
                num_iters=self.config.num_iters,
                current_client_progress=0,  # local training
                selected_clients_count=len(self._clients),
                training_phase="training",
            )
        with tqdm(
            range(self.config.num_iters),
            total=self.config.num_iters,
            desc=f"{self.config.algorithm} training",
            unit="iter",
            mininterval=1,  # usually useless since a full iteration takes a very long time
        ) as outer_pbar:
            for self.n_iter in outer_pbar:
                with tqdm(
                    total=len(self._clients),
                    desc=f"Iter {self.n_iter+1}/{self.config.num_iters}",
                    unit="client",
                    mininterval=max(1, len(self._clients) // 10),
                    disable=self.config.verbose < 1,
                    leave=False,
                ) as pbar:
                    for client_id in range(len(self._clients)):
                        client = self._clients[client_id]
                        client.train()
                        if self.n_iter > 0 and (self.n_iter + 1) % self.config.eval_every == 0:
                            part = "train"  # only evaluate train metrics on clients
                            metrics = client.evaluate(part)
                            self._logger_manager.log_metrics(
                                client_id,
                                metrics,
                                step=self.n_iter,
                                epoch=self.n_iter,
                                part=part,
                            )
                        pbar.update(1)
                        # report progress for multi-processing
                        if self._subprocess:
                            self._report_progress(
                                n_iter=self.n_iter + 1,
                                num_iters=self.config.num_iters,
                                current_client_progress=client_id + 1,  # counts the number of processed clients
                                selected_clients_count=len(self._clients),
                                training_phase="training",
                            )
        self._logger_manager.log_message("Local training finished...")
        self._logger_manager.flush()
        self._complete_experiment = True

    def pre_iter(self) -> None:
        """Preprocess before each iteration, after evaluation.
        Can be used to update the model, e.g. update the local model.

        Returns
        -------
        None
        """
        pass

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
        for X, y in dataloader:
            X, y = X.to(self.model.device), y.to(self.model.device)
            probs = self.model(X)
            metrics.append(self.dataset.evaluate(probs, y))
        num_samples = sum([m["num_samples"] for m in metrics])
        metrics_names = [k for k in metrics[0] if k != "num_samples"]
        metrics = {k: sum([m[k] * m["num_samples"] for m in metrics]) / num_samples for k in metrics_names}
        metrics["num_samples"] = num_samples
        # free memory
        del X, y, probs
        return metrics

    def aggregate_client_metrics(self, ignore: Sequence[str] = None) -> None:
        """Aggregate the metrics transmitted from the clients.

        Parameters
        ----------
        ignore : Sequence[str], optional
            The metrics to ignore.

        Returns
        -------
        None

        """
        ignore = ignore or []
        if not any(["metrics" in m for m in self._received_messages]):
            raise ValueError("no metrics received from clients")
        # cache metrics stored in self._metrics into self._cached_metrics
        if self._metrics:  # not empty
            self._cached_metrics.append(self._metrics.copy())
        new_metrics = defaultdict(lambda: defaultdict(float))
        part = "train"  # only aggregate train metrics
        assert part in self.dataset.data_parts, f"Invalid part name {part}, should be one of {self.dataset.data_parts}."
        for m in self._received_messages:
            if "metrics" not in m:
                continue
            for k, v in m["metrics"][part].items():
                if k != "num_samples":
                    new_metrics[part][k] += m["metrics"][part][k] * m["metrics"][part]["num_samples"]
                elif k in ignore:
                    continue
                else:  # num_samples
                    new_metrics[part][k] += m["metrics"][part][k]
        for k in new_metrics[part]:
            if k != "num_samples":
                new_metrics[part][k] /= new_metrics[part]["num_samples"]
        self._logger_manager.log_metrics(
            None,
            default_dict_to_dict(new_metrics[part]),
            step=self.n_iter + 1,
            epoch=self.n_iter + 1,
            part=part,
        )
        # move self._metrics to self._cached_metrics and refresh self._metrics
        self._cached_metrics.append(self._metrics.copy())
        self._metrics = default_dict_to_dict(new_metrics)
        del new_metrics
        # TODO: use the cached sequence `self._cached_metrics` of aggregated metrics
        # to update the status varibles `self._is_convergent`

    def add_parameters(self, params: Iterable[Parameter], ratio: float) -> None:
        """Update the server's parameters with the given parameters.

        Parameters
        ----------
        params : Iterable[torch.nn.parameter.Parameter]
            The parameters to be added.
        ratio : float
            The ratio of the parameters to be added.

        Returns
        -------
        None

        """
        for server_param, param in zip(self.model.parameters(), params):
            server_param.data.add_(param.data.detach().clone().to(self.device), alpha=ratio)

    def avg_parameters(self, size_aware: bool = False, inertia: float = 0.0) -> None:
        """Update the server's parameters via
        averaging the parameters received from the clients.

        Parameters
        ----------
        size_aware : bool, default False
            Whether to use the size-aware averaging,
            which is the weighted average of the parameters,
            where the weight is the number of training samples.
            From the view of optimization theory,
            this is recommended to be set `False`.
        inertia : float, default 0.0
            The weight of the previous parameters,
            should be in the range [0, 1).

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
            ratio = (m["train_samples"] / total_samples if size_aware else 1 / len(self._received_messages)) * (1 - inertia)
            self.add_parameters(m["parameters"], ratio)

    def update_gradients(self) -> None:
        """Update the server's gradients."""
        if len(self._received_messages) == 0:
            return
        assert all(["gradients" in m for m in self._received_messages]), "some clients have not sent gradients yet"
        # self.model.zero_grad()
        for mp, gd in zip(self.model.parameters(), self._received_messages[0]["gradients"]):
            mp.grad = torch.zeros_like(gd).to(self.device)
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for rm in self._received_messages:
            # TODO: consider alpha = 1 / len(self._received_messages)
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

    def get_cached_metrics(self, client_idx: Optional[int] = None) -> List[Dict[str, float]]:
        """Get the cached metrics of the given client,
        or the cached aggregated metrics stored on the server.

        Parameters
        ----------
        client_idx : int, optional
            The index of the client. If `None`,
            returns the cached aggregated metrics stored on the server.

        Returns
        -------
        List[Dict[str, float]]
            The cached metrics of the given client,
            or the cached aggregated metrics stored on the server.

        """
        if client_idx is None:
            return self._cached_metrics
        if client_idx >= len(self._clients):
            raise ValueError(f"client_idx should be less than {len(self._clients)}")

        return self._clients[client_idx]._cached_metrics

    def _reset(self, reset_clients: bool = True) -> None:
        """Reset the server.

        State variables are reset to the initial state,
        and the logger manager is reset, which outputs
        to new log files.

        Parameters
        ----------
        reset_clients : bool, default True
            Whether to reset the clients.

        Returns
        -------
        None

        """
        self._received_messages = []
        self._metrics = {}
        self._cached_metrics = []
        self._is_convergent = False
        self.n_iters = 0
        self._num_communications = 0
        self._logger_manager.reset()
        self._complete_experiment = False
        # reset the global model
        reset_parameters(self.model)
        # reset the clients
        if reset_clients:
            for c in tqdm(self._clients, desc="Resetting clients", mininterval=1, leave=False):
                c._reset()

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

    @property
    def is_convergent(self) -> bool:
        """Whether the training process is convergent."""
        return self._is_convergent
