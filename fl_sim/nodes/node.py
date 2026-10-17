"""
The base Node class.
====================

"""

import types
import warnings
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch_ecg.utils import ReprMixin

from ..utils.misc import aggregate_results_from_json_log
from ..utils.torch_compat import torch_norm

__all__ = [
    "Node",
]


class Node(ReprMixin, ABC):
    """An abstract base class for the server and client nodes."""

    __name__ = "Node"

    @abstractmethod
    def communicate(self, target: "Node") -> None:
        """Communicate with the target node.

        The current node communicates model parameters, gradients, etc. to `target` node.

        The **typed API** (recommended) sends
        :class:`~fl_sim.messages.ClientToServer` / :class:`~fl_sim.messages.ServerToClient`
        messages via :meth:`Server.receive_one` / :meth:`Client.receive`; the base classes
        provide working default implementations for the common cases:

        .. code-block:: python

            # client -> server (typed, default implementation of Client.communicate)
            target.receive_one(
                ClientToServer(
                    client_id=self.client_id,
                    train_samples=...,
                    metrics=self._metrics,
                    parameters=self.get_detached_model_parameters(),
                )
            )
            # server -> client (typed, default implementation of Server.communicate)
            target.receive(ServerToClient(parameters=self.get_detached_model_parameters()))

        The **legacy API** (kept working, used by the built-in algorithms) manipulates
        the ``_received_messages`` containers directly:

        .. code-block:: python

            # client -> server (legacy)
            target._received_messages.append(
                ClientMessage(
                    {
                        "client_id": self.client_id,
                        "parameters": self.get_detached_model_parameters(),
                        "train_samples": ...,
                        "metrics": self._metrics,
                    }
                )
            )
            # server -> client (legacy)
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
        """Extension point for subclasses;
        called at the end of `Server.__init__` and `Client.__init__`,
        after the configs are set.
        """
        # NOTE: required config fields are enforced by the (pydantic) configs
        # themselves at construction time, so no check is needed here.

    @property
    @abstractmethod
    def is_convergent(self) -> bool:
        """Whether the training process on the node is convergent."""
        raise NotImplementedError

    def get_detached_model_parameters(self) -> List[Tensor]:
        """Get the detached model parameters."""
        return [p.detach().clone() for p in self.model.parameters()]

    def compute_gradients(
        self,
        at: Optional[Sequence[Tensor]] = None,
        dataloader: Optional[DataLoader] = None,
    ) -> List[Tensor]:
        """Compute the gradients of the model on the node.

        The gradients are computed on the model parameters `at` or the current model parameters,
        as the average of the gradients on the mini-batches from `dataloader` or `self.train_loader`.

        Parameters
        ----------
        at : list of torch.Tensor, optional
            The model parameters to compute the gradients.
            None for the current model parameters.
        dataloader : torch.utils.data.DataLoader, optional
            The dataloader to compute the gradients.
            None for `self.train_loader`.

        """
        prev_training = self.model.training
        prev_model_state_dict = self.model.state_dict()
        if at is not None:
            self.set_parameters(at)
        if dataloader is None:
            assert hasattr(self, "train_loader") and self.train_loader is not None, "train_loader is not set"
            dataloader = self.train_loader
        assert len(dataloader) > 0, "empty dataloader"

        self.model.train()
        self.optimizer.zero_grad()
        total_samples = len(dataloader.dataset)
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y) * len(X) / total_samples
            # gradients are accumulated after each backward pass
            # one should **NOT** call `optimizer.zero_grad()` before the backward pass
            # and should **NOT** call `optimizer.step()` after the backward pass
            # since the model parameters should not be updated
            # otherwise, the result is not the gradients **at** ``at``
            loss.backward()
        mini_batch_grads = [p.grad.detach().clone() for p in self.model.parameters()]
        # set the model state back and clear the gradients
        self.model.load_state_dict(prev_model_state_dict)
        self.optimizer.zero_grad()
        self.model.train(prev_training)
        # garbage collection
        del prev_model_state_dict

        return mini_batch_grads

    def get_gradients(
        self,
        norm: Optional[Union[str, int, float]] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> Union[float, List[Tensor]]:
        """Get the gradients or norm of the gradients
        of the model on the node.

        Parameters
        ----------
        norm : str or int or float, optional
            The norm of the gradients to compute.
            None for the raw gradients (list of tensors).
            Refer to :func:`torch.linalg.norm` for more details.
        model : torch.nn.Module, optional
            The model to get the gradients,
            default to `self.model`.

        Returns
        -------
        float or List[torch.Tensor]
            The gradients or norm of the gradients.

        """
        if model is None:
            model = self.model
        grads = []
        for param in model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        if norm is not None:
            if len(grads) == 0:
                grads = 0.0
                warnings.warn("No gradients available. Set to 0.0 by default.", RuntimeWarning)
            else:
                grads = torch_norm(torch.cat([grad.view(-1) for grad in grads]), norm).item()
        return grads

    @staticmethod
    def get_norm(
        tensor: Union[
            Number,
            Tensor,
            np.ndarray,
            Parameter,
            types.GeneratorType,
            Sequence[Union[np.ndarray, Tensor, Parameter, types.GeneratorType]],
        ],
        norm: Union[str, int, float] = "fro",
    ) -> float:
        """Get the norm of a tensor.

        Parameters
        ----------
        tensor : torch.Tensor or np.ndarray or torch.nn.parameter.Parameter or generator or list
            The tensor (array, parameter, etc.) to compute the norm.
        norm : str or int or float, default "fro"
            The norm to compute.
            Refer to :func:`torch.linalg.norm` for more details.

        Returns
        -------
        float
            The norm of the tensor.

        """
        if tensor is None:
            return float("nan")
        if isinstance(tensor, Number):
            return tensor
        if isinstance(tensor, Tensor):
            tensor = [tensor]
        elif isinstance(tensor, np.ndarray):
            tensor = [torch.from_numpy(tensor)]
        elif isinstance(tensor, Parameter):
            tensor = [tensor.data]
        elif isinstance(tensor, (list, tuple)):
            tensor = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t.detach().clone() for t in tensor]
        elif isinstance(tensor, types.GeneratorType):
            return Node.get_norm(list(tensor), norm)
        else:
            raise TypeError(f"Unsupported type: {type(tensor)}")
        return torch_norm(torch.cat([t.view(-1) for t in tensor]), norm).item()

    def set_parameters(self, params: Iterable[Parameter], model: Optional[torch.nn.Module] = None) -> None:
        """Set the parameters of the model on the node.

        Parameters
        ----------
        params : Iterable[torch.nn.parameter.Parameter]
            The parameters to set.
        model : torch.nn.Module, optional
            The model to set the parameters,
            default to `self.model`.

        Returns
        -------
        None

        """
        if model is None:
            model = self.model
        for node_param, param in zip(model.parameters(), params):
            node_param.data = param.data.detach().clone().to(self.device)

    @staticmethod
    def aggregate_results_from_json_log(d: Union[dict, str, Path], part: str = "val", metric: str = "acc") -> np.ndarray:
        """Aggregate the federated results from json log.

        Thin static-method wrapper kept for backward compatibility;
        see :func:`fl_sim.utils.misc.aggregate_results_from_json_log`
        for the (maintained) implementation and the expected log structure.
        """
        return aggregate_results_from_json_log(d, part=part, metric=metric)
