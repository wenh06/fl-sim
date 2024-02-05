"""
Based on the paper, serving as the baseline for LCFL.

`An Efficient Framework for Clustered Federated Learning. <https://arxiv.org/abs/2102.04803>`_

Codebase URL: https://github.com/jichan3751/ifca

"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Sequence

import torch
from torch_ecg.utils.misc import add_docstring, list_sum

from ...nodes import ClientMessage
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm
from ..fedopt import FedAvgClient as BaseClient
from ..fedopt import FedAvgClientConfig as BaseClientConfig
from ..fedopt import FedAvgServer as BaseServer
from ..fedopt import FedAvgServerConfig as BaseServerConfig

__all__ = [
    "IFCAClient",
    "IFCAClientConfig",
    "IFCAServer",
    "IFCAServerConfig",
]


_base_algorithm = "FedAvg"


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class IFCAServerConfig(BaseServerConfig):
    """Server config for the IFCA algorithm.

    Parameters
    ----------
    num_clusters : int
        The number of clusters.
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float, default 1
        The ratio of clients to participate in each round.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "IFCAServerConfig"

    def __init__(
        self,
        num_clusters: int,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float = 1,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ServerConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio=clients_sample_ratio,
            **kwargs,
        )
        self.algorithm = name
        self.num_clusters = num_clusters


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class IFCAClientConfig(BaseClientConfig):
    """Client config for the IFCA algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "IFCAClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ClientConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            **kwargs,
        )
        self.algorithm = name


@register_algorithm()
@add_docstring(BaseServer.__doc__.replace(_base_algorithm, "IFCA"))
class IFCAServer(BaseServer):
    __name__ = "IFCAServer"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        check compatibility of server and client configs,
        and set cluster centers
        """
        super()._post_init()
        assert self.config.num_clusters > 0
        self._cluster_centers = {
            cluster_id: {
                "center_model_params": [p.detach().clone() for p in self.model.parameters()],
                "client_ids": [],
            }
            for cluster_id in range(self.config.num_clusters)
        }

    @property
    def client_cls(self) -> type:
        return IFCAClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": IFCAServerConfig,
            "client": IFCAClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return ["num_clusters"]

    def communicate(self, target: "IFCAClient") -> None:
        """Send cluster centers to client"""
        target._received_messages = {
            "cluster_centers": {
                cluster_id: deepcopy(cluster["center_model_params"]) for cluster_id, cluster in self._cluster_centers.items()
            }
        }

    @torch.no_grad()
    def update(self) -> None:
        """Update cluster centers"""
        # cache the client ids of each cluster of the previous iteration
        prev_client_ids = {cluster_id: deepcopy(cluster["client_ids"]) for cluster_id, cluster in self._cluster_centers.items()}
        # reset the list of client ids of each cluster
        for cluster_id, cluster in self._cluster_centers.items():
            cluster["client_ids"] = []
        # check the size of each cluster from the received messages
        cluster_sizes = {cluster_id: 0 for cluster_id in range(self.config.num_clusters)}
        for m in self._received_messages:
            cluster_sizes[m["cluster_id"]] += 1
            self._cluster_centers[m["cluster_id"]]["client_ids"].append(m["client_id"])
        # if a client in some cluster does not participate in this round,
        # add it back to the cluster
        collected_client_ids = list_sum((cluster["client_ids"] for cluster in self._cluster_centers.values()))
        for cluster_id, cluster in self._cluster_centers.items():
            for client_id in prev_client_ids[cluster_id]:
                if client_id not in collected_client_ids:
                    cluster["client_ids"].append(client_id)
        # update the cluster centers via averaging the delta_parameters from each client
        for m in self._received_messages:
            cluster_id = m["cluster_id"]
            cluster = self._cluster_centers[cluster_id]
            for p, p_delta in zip(cluster["center_model_params"], m["delta_parameters"]):
                p.data.add_(
                    p_delta.data.detach().clone().to(self.device),
                    alpha=1 / cluster_sizes[cluster_id],
                )
            cluster["client_ids"].append(m["client_id"])

    def aggregate_client_metrics(self, ignore: Sequence[str] = ["cluster_id"]) -> None:
        """Aggregate the metrics transmitted from the clients.

        Parameters
        ----------
        ignore : Sequence[str], default ["cluster_id"]
            The metrics to ignore.

        Returns
        -------
        None

        """
        super().aggregate_client_metrics(ignore=ignore)


@register_algorithm()
@add_docstring(BaseClient.__doc__.replace(_base_algorithm, "IFCA"))
class IFCAClient(BaseClient):
    __name__ = "IFCAClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        self.cluster_id = -1

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "IFCAServer") -> None:
        delta_parameters = self.get_detached_model_parameters()
        for dp, rp in zip(delta_parameters, self._cached_parameters):
            dp.data.add_(rp.data, alpha=-1)
        message = {
            "client_id": self.client_id,
            "cluster_id": self.cluster_id,
            "delta_parameters": delta_parameters,
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        """Perform clustering and local training."""
        losses = {}
        # cache current model parameters and metrics
        local_model_weights = self.model.state_dict()
        prev_metrics = self._metrics.copy()
        with torch.no_grad():
            for cluster_id, cluster in self._received_messages["cluster_centers"].items():
                # load cluster center into self.model
                for p, p_center in zip(self.model.parameters(), cluster):
                    p.data.copy_(p_center.data)
                # evaluate the loss
                losses[cluster_id] = self.evaluate(part="train")["loss"]
        # restore model parameters and metrics
        self.model.load_state_dict(local_model_weights)
        self._metrics = prev_metrics.copy()
        del local_model_weights, prev_metrics
        # select the cluster with the minimum loss
        self.cluster_id = min(losses, key=losses.get)

        # set the cluster center as the local model parameters
        self._cached_parameters = [
            p.detach().clone().to(self.device) for p in self._received_messages["cluster_centers"][self.cluster_id]
        ]
        self.set_parameters(self._cached_parameters)
        self.solve_inner()  # alias of self.train()
