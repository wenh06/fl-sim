"""
The base Client class.
======================

"""

import warnings
from abc import abstractmethod
from copy import deepcopy
from inspect import signature
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_ecg.utils import add_docstring

from ..data_processing.fed_dataset import FedDataset
from ..messages import ClientToServer, ServerToClient
from ..models import reset_parameters
from ..optimizers import get_optimizer
from ..utils.misc import get_scheduler
from .configs import ClientConfig
from .node import Node

if TYPE_CHECKING:
    from .server import Server

__all__ = [
    "Client",
]


def _communicate_accepts_target(communicate) -> bool:
    """Whether a ``communicate`` hook is written in the deprecated legacy
    style ``communicate(self, target)`` (as opposed to the current
    ``communicate(self)``)."""
    try:
        positional = [
            param
            for param in signature(communicate).parameters.values()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):  # builtins etc. — treat as new-style
        return False
    # NOTE: ``communicate`` is a bound method here, so ``self`` is already
    # excluded from the signature: any remaining positional parameter is the
    # (legacy) server argument.
    return len(positional) >= 1


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

        self.train_loader, self.val_loader = self.dataset.get_dataloader(
            self.config.batch_size, self.config.batch_size, self.client_id
        )

        self.optimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            params=self.model.parameters(),
            config=self.config,
        )
        scheduler_config = {k: v for k, v in self.config.scheduler.items() if k != "name"}
        self.lr_scheduler = get_scheduler(
            scheduler_name=self.config.scheduler["name"],
            optimizer=self.optimizer,
            config=scheduler_config,
        )

        self._cached_parameters = None
        self._received_messages = {}
        # the latest typed message received from the server
        # (see :meth:`Client.receive`), `None` if the legacy
        # `Server.communicate` (writing `_received_messages` directly) is used
        self._latest_message = None
        self._metrics = {}
        self._cached_metrics = []  # container for caching metrics
        self._is_convergent = False

        self._post_init()

        # checks that the client has all the required attributes in config.extra_observes
        for attr in self.config.extra_observes:
            assert hasattr(self, attr), f"{self.__name__} should have attribute {attr} for extra observes."

    # round flags written by ``Server._decide_uplink`` (read-only for the
    # client's own hooks); default: every selected client communicates and
    # is counted (the historical behavior of all algorithms except
    # FedPD/ProxSkip, whose servers override ``_decide_uplink``).
    _communicate_this_round: bool = True
    _count_if_skipped: bool = True
    _round_idx: Optional[int] = None

    def build_reply(self, target: Optional["Server"] = None) -> Optional[ClientToServer]:
        """Build the (typed) reply message to be uploaded to the server.

        This is the pure-client part of the former ``_communicate``: validate
        the metrics, let the (algorithm-overridable) :meth:`communicate` hook
        build the message, and archive the metrics. It does NOT touch the
        server (the invariant of the parallel execution layer), so it can run
        inside a worker task.

        Parameters
        ----------
        target : Server, optional
            The server, required only by legacy-style (deprecated)
            ``communicate(self, target)`` overrides that upload by
            themselves. New-style hooks ignore it.

        Returns
        -------
        ClientToServer or None
            The reply message; ``None`` if the client does not communicate
            this round (decided by ``_communicate_this_round``).

        """
        # check validity of self._metrics
        for part, metrics in self._metrics.items():
            assert isinstance(metrics, dict), f"metrics for {part} should be a dict, " f"but got {type(metrics).__name__}"
            assert "num_samples" in metrics, (
                "In order to let the server aggregate the metrics, "
                f"metrics for {part} should have key `num_samples`, "
                f"but got {metrics.keys()}"
            )
        reply: Optional[ClientToServer] = None
        if self._communicate_this_round:
            if _communicate_accepts_target(self.communicate):
                if target is None:
                    raise TypeError(
                        f"{type(self).__name__}.communicate still accepts the server argument; "
                        "legacy-style communicate overrides only support the sequential backend "
                        "(they can read server state, which no parallel backend allows)."
                    )
                warnings.warn(
                    f"{type(self).__name__}.communicate(self, target) is deprecated; "
                    "override communicate(self) and return the ClientToServer message instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                reply = self.communicate(target)
            else:
                reply = self.communicate()
        # move self._metrics to self._cached_metrics
        self._cached_metrics.append(self._metrics.copy())
        self._metrics = {}  # clear the metrics
        return reply

    def _communicate(self, target: "Server") -> None:
        """Build the reply via :meth:`build_reply` and upload it to the server,
        maintaining the communication counter.

        A client that does not communicate this round still counts, unless
        its ``_count_if_skipped`` flag is set to ``False`` by the server's
        ``_decide_uplink`` (FedPD is the only such algorithm).

        Parameters
        ----------
        target : Server
            The server to communicate with.

        Returns
        -------
        None

        """
        reply = self.build_reply(target)
        if reply is not None:
            target.receive_one(reply)
            target._num_communications += 1
        elif self._count_if_skipped:
            target._num_communications += 1

    def accept_parameters(self) -> None:
        """
        Accepts parameters from received_messages.

        The named ``"model_state"`` mapping (the recommended exchange format,
        see :data:`~fl_sim.messages.ModelState`) takes precedence over the
        flat positional ``"parameters"`` list.

        Need to be implemented in the child class to specify part in received_messages
        if the key is not "parameters"/"model_state".

        Some algorithms (e.g. ``IFCA``, ``FedPD``) do not put the parameters under
        the ``"parameters"`` key of the received messages (or communicate only
        with a subset of clients in a round); for those, the parameters are
        accepted in the client's :meth:`update`, so a missing key is not an error.
        """
        if "model_state" in self._received_messages:
            self.set_model_state(self._received_messages["model_state"])
        elif "parameters" in self._received_messages:
            self.set_parameters(self._received_messages["parameters"])

    def receive(self, message: ServerToClient) -> None:
        """Receive a typed message broadcast from the server.

        The message is stored in ``self._latest_message`` (for typed-API
        consumers) and merged into the legacy ``self._received_messages``
        dict (for legacy consumers, e.g. :meth:`accept_parameters`).

        Parameters
        ----------
        message : ServerToClient
            The typed message broadcast from the server.

        Returns
        -------
        None

        """
        self._latest_message = message
        self._received_messages = message.to_legacy_dict()

    def communicate(self) -> ClientToServer:
        """Build the local model parameters and metrics as the reply message.

        This is the default (typed-API) implementation; subclasses override it
        for algorithm-specific payloads (deltas, gradients, control variates,
        etc.). The hook must NOT touch the server: it builds and returns the
        :class:`~fl_sim.messages.ClientToServer`; the upload (and the
        communication counter) is handled by :meth:`_communicate`. This
        purity is what lets the hook run inside a parallel worker task.

        Returns
        -------
        ClientToServer
            The reply message to upload.

        """
        return ClientToServer(
            client_id=self.client_id,
            train_samples=len(self.train_loader.dataset),
            metrics=self._metrics,
            parameters=self.get_detached_model_parameters(),
        )

    def _update(self) -> None:
        """Client update, and clear cached messages
        from the server of the previous iteration.
        """
        self.update()
        # free the memory in received_messages
        del self._received_messages
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
            self.lr_scheduler.step()

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
        assert part in self.dataset.data_parts, f"Invalid part name, should be one of {self.dataset.data_parts}."
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
        # get the gradient norm of the local model
        self._metrics[part]["grad_norm"] = self.get_gradients(norm="fro")
        # record items in config.extra_observes
        for attr in self.config.extra_observes:
            self._metrics[part][attr] = Node.get_norm(getattr(self, attr))
        # free the memory
        del all_logits, all_labels, X, y
        return self._metrics[part]

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

    def _reset(self) -> None:
        """Reset the model and optimizer."""
        # reset the model
        reset_parameters(self.model)
        # reset the optimizer and clear gradients
        del self.optimizer
        self.optimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            params=self.model.parameters(),
            config=self.config,
        )
        self.optimizer.zero_grad()

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "client_id",
            "config",
        ]

    @property
    def is_convergent(self) -> bool:
        """Whether the training process is convergent."""
        return self._is_convergent
