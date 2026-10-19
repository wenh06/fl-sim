"""
Based on the paper, serving as the baseline for LCFL.

`An Efficient Framework for Clustered Federated Learning. <https://arxiv.org/abs/2102.04803>`_

Codebase URL: https://github.com/jichan3751/ifca

"""

from copy import deepcopy
from typing import Dict, Sequence

import torch
from torch_ecg.utils.misc import add_docstring, list_sum

from ...messages import ClientToServer, ServerToClient
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


def _named_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """All model parameters as a named tensor mapping
    (the :data:`~fl_sim.messages.ModelState` format; parameters only, no buffers)."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


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

    num_clusters: int
    clients_sample_ratio: float = 1


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

    lr: float = 1e-2


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
                "center_model_state": _named_params(self.model),
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

    def communicate(self, target: "IFCAClient") -> None:
        """Send the cluster centers (named model states) to the client."""
        target.receive(
            ServerToClient(
                extras={
                    "cluster_centers": {
                        cluster_id: deepcopy(cluster["center_model_state"])
                        for cluster_id, cluster in self._cluster_centers.items()
                    }
                }
            )
        )

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
        # update the cluster centers via averaging the delta (model) states from each client, by name
        for m in self._received_messages:
            cluster_id = m["cluster_id"]
            cluster = self._cluster_centers[cluster_id]
            for n, p_delta in m["model_state"].items():
                cluster["center_model_state"][n].data.add_(
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

    def _load_named_params(self, state: Dict[str, torch.Tensor]) -> None:
        """Load a (parameters-only) named mapping into the model, by name."""
        for n, p in self.model.named_parameters():
            p.data.copy_(state[n].data.to(p.device))

    def communicate(self, target: "IFCAServer") -> None:
        # the delta (model) state: current parameters minus the chosen cluster center, by name
        delta_state = _named_params(self.model)
        for n, v in delta_state.items():
            v.data.add_(self._cached_parameters[n].data.to(v.device), alpha=-1)
        target.receive_one(
            ClientToServer(
                client_id=self.client_id,
                train_samples=len(self.train_loader.dataset),
                metrics=self._metrics,
                model_state=delta_state,
                extras={"cluster_id": self.cluster_id},
            )
        )

    def update(self) -> None:
        """Perform clustering and local training."""
        losses = {}
        # cache current model parameters and metrics
        local_model_weights = self.model.state_dict()
        prev_metrics = self._metrics.copy()
        with torch.no_grad():
            for cluster_id, center in self._received_messages["cluster_centers"].items():
                # load the cluster center into self.model (by name)
                self._load_named_params(center)
                # evaluate the loss
                losses[cluster_id] = self.evaluate(part="train")["loss"]
        # restore model parameters and metrics
        self.model.load_state_dict(local_model_weights)
        self._metrics = prev_metrics.copy()
        del local_model_weights, prev_metrics
        # select the cluster with the minimum loss
        self.cluster_id = min(losses, key=losses.get)

        # set the chosen cluster center as the local model parameters
        self._cached_parameters = deepcopy(self._received_messages["cluster_centers"][self.cluster_id])
        self._load_named_params(self._cached_parameters)
        self.solve_inner()  # alias of self.train()
