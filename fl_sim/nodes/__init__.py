"""
fl_sim.nodes
=============

:class:`Node` s are the core of the simulation framework.
:class:`Node` has two subclasses: :class:`Server` and :class:`Client`.
The :class:`Server` class is the base class for all servers,
which acts as the coordinator of the training process, as well as maintainer of status variables.
The :class:`Client` class is the base class for all clients.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: fl_sim.nodes

Base Node class
---------------
.. autosummary::
    :toctree: generated/

    Node

Server classes
--------------
.. autosummary::
    :toctree: generated/

    Server
    ServerConfig

Client classes
--------------
.. autosummary::
    :toctree: generated/

    Client
    ClientConfig

"""

from ..messages import ClientMessage, ClientToServer, ModelState, ServerToClient
from .client import Client
from .configs import ClientConfig, ServerConfig
from .node import Node
from .server import Server

__all__ = [
    "Server",
    "Client",
    "ServerConfig",
    "ClientConfig",
    "ClientMessage",
    "ServerToClient",
    "ClientToServer",
    "ModelState",
    "Node",
]
