"""
fl_sim.messages
===============

Typed messages exchanged between the :class:`~fl_sim.nodes.Server` and
:class:`~fl_sim.nodes.Client` nodes.

The legacy protocol manipulates the (private) ``_received_messages``
containers directly with magic string keys, e.g.

.. code-block:: python

    # server -> client (legacy)
    target._received_messages = {"parameters": params, "alpha": 0.01}
    # client -> server (legacy)
    target._received_messages.append(
        ClientMessage(client_id=..., train_samples=..., metrics=...)
    )

The typed messages below are the recommended replacement: the message
fields are explicit (hence IDE-discoverable and validated by construction),
and the (private) legacy containers are populated automatically by
:meth:`Client.receive` and :meth:`Server.receive_one`, so that the legacy
consumption code (e.g. :meth:`Client.accept_parameters`) keeps working
unchanged during the transition.

.. code-block:: python

    # server -> client (typed)
    target.receive(ServerToClient(parameters=params, extras={"alpha": 0.01}))
    # client -> server (typed)
    target.receive_one(ClientToServer(client_id=..., train_samples=..., metrics=...))

.. contents::
    :depth: 2
    :local:
    :backlinks: top

"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from torch import Tensor

# The (recommended) named-parameter exchange format: an ordered mapping
# from component names to (general) tensors. The model itself does NOT
# need to be a torch.nn.Module — a linear model's ``{"w": ..., "b": ...}``
# or a matrix factorization's ``{"U": ..., "V": ...}`` are first-class.
# For torch modules this is exactly what ``state_dict`` produces, and the
# legacy flat ``List[Tensor]`` maps onto it with positional names
# ("0", "1", ...) via :meth:`fl_sim.nodes.Node.get_model_state`.
ModelState = Dict[str, Tensor]

__all__ = [
    "ClientToServer",
    "ServerToClient",
    "ClientMessage",
    "ModelState",
]


@dataclass
class ServerToClient:
    """The (typed) message broadcast from the server to a client.

    Parameters
    ----------
    parameters : List[torch.Tensor], optional
        The (flat, positional) model parameters broadcast to the client.
        `None` for algorithms that do not broadcast parameters
        (e.g. sending gradients or other statistics instead),
        in which case the payload goes to `extras`.
    model_state : Mapping[str, torch.Tensor], optional
        The named model parameters broadcast to the client,
        see :data:`ModelState`. Preferred over the flat `parameters`
        for models with semantically distinct components
        (e.g. encoder/head splits), or for non-module models.
        May coexist with `parameters`; consumers preferring the named
        format handle it first (see :meth:`Client.accept_parameters`).
    extras : Dict[str, Any], optional
        Algorithm-specific payloads, e.g. hyperparameters
        (``{"alpha": 0.01}``), statistics (``{"global_stats": ...}``), etc.

    """

    parameters: Optional[List[Tensor]] = None
    model_state: Optional[Mapping[str, Tensor]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert the typed message to the legacy ``_received_messages`` dict."""
        d: Dict[str, Any] = dict(self.extras)
        if self.parameters is not None:
            d["parameters"] = self.parameters
        if self.model_state is not None:
            d["model_state"] = dict(self.model_state)
        return d


@dataclass
class ClientToServer:
    """The (typed) message uploaded from a client to the server.

    Parameters
    ----------
    client_id : int
        The ID of the client sending the message.
    train_samples : int
        The number of training samples on the client.
    metrics : Dict[str, Any]
        The (part-nested) evaluated metrics of the client,
        e.g. ``{"train": {"acc": ..., "loss": ..., "num_samples": ...}}``.
    parameters : List[torch.Tensor], optional
        The (flat, positional) parameters (or deltas, gradients, etc.)
        uploaded by the client.
        `None` for algorithms that upload nothing parameter-like.
    model_state : Mapping[str, torch.Tensor], optional
        The named model parameters uploaded by the client,
        see :data:`ModelState`. Preferred over the flat `parameters`
        for models with semantically distinct components, or for
        non-module models. May coexist with `parameters`.
    extras : Dict[str, Any], optional
        Algorithm-specific payloads, e.g. control variates
        (``{"control_variates": ...}``), cluster ids, etc.

    """

    client_id: int
    train_samples: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    parameters: Optional[List[Tensor]] = None
    model_state: Optional[Mapping[str, Tensor]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        """Convert the typed message to the legacy :class:`~fl_sim.nodes.ClientMessage` kwargs."""
        d: Dict[str, Any] = dict(self.extras)
        if self.parameters is not None:
            d["parameters"] = self.parameters
        if self.model_state is not None:
            d["model_state"] = dict(self.model_state)
        d.update(
            client_id=self.client_id,
            train_samples=self.train_samples,
            metrics=self.metrics,
        )
        return d


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

    def __init__(self, client_id: int, train_samples: int, metrics: dict, **kwargs) -> None:
        super().__init__(client_id=client_id, train_samples=train_samples, metrics=metrics, **kwargs)

    @classmethod
    def from_typed(cls, message: ClientToServer) -> "ClientMessage":
        """Build the legacy message from the typed :class:`ClientToServer`."""
        return cls(**message.to_legacy_kwargs())
