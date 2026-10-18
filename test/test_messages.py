""" """

from types import SimpleNamespace
from typing import List

import torch
from torch import Tensor

from fl_sim.messages import ClientToServer, ServerToClient
from fl_sim.nodes import Client, ClientMessage, Server


def _fake_client() -> SimpleNamespace:
    """A namespace carrying only the client attributes that the tested
    methods touch; the methods are invoked unbound (they are plain logic).
    """
    ns = SimpleNamespace(
        _received_messages={},
        _latest_message=None,
        _cached_parameters=None,
    )
    ns.set_parameters = lambda params, model=None: setattr(ns, "_cached_parameters", params)
    return ns


def _fake_server() -> SimpleNamespace:
    return SimpleNamespace(_received_messages=[])


def _params() -> List[Tensor]:
    return [torch.arange(6.0).reshape(2, 3)]


def test_server_to_client_legacy_dict():
    msg = ServerToClient(parameters=_params(), extras={"alpha": 0.01, "beta": 2})
    d = msg.to_legacy_dict()
    assert d["parameters"] == msg.parameters
    assert d["alpha"] == 0.01 and d["beta"] == 2

    # no parameters -> the key is absent (legacy "missing key is not an error" semantics)
    d2 = ServerToClient(extras={"global_stats": {"c": (1, 2)}}).to_legacy_dict()
    assert "parameters" not in d2 and "global_stats" in d2


def test_client_receive_typed_and_legacy_views():
    client = _fake_client()
    msg = ServerToClient(parameters=_params(), extras={"alpha": 0.01})
    Client.receive(client, msg)
    # typed view
    assert client._latest_message is msg
    # legacy view (e.g. for accept_parameters)
    assert "parameters" in client._received_messages
    assert client._received_messages["alpha"] == 0.01


def test_client_to_server_conversion():
    typed = ClientToServer(
        client_id=3,
        train_samples=100,
        metrics={"train": {"acc": 0.5, "num_samples": 100}},
        parameters=_params(),
        extras={"control_variates": [torch.zeros(2, 3)]},
    )
    legacy = ClientMessage.from_typed(typed)
    assert legacy["client_id"] == 3
    assert legacy["train_samples"] == 100
    assert legacy["metrics"]["train"]["acc"] == 0.5
    assert legacy["parameters"] == typed.parameters
    assert legacy["control_variates"] == typed.extras["control_variates"]

    # no parameters -> the key is absent
    legacy2 = ClientMessage.from_typed(ClientToServer(client_id=1, train_samples=2, metrics={}))
    assert "parameters" not in legacy2


def test_server_receive_one():
    server = _fake_server()
    typed = ClientToServer(client_id=7, train_samples=10, metrics={"train": {"num_samples": 10}})
    Server.receive_one(server, typed)
    assert len(server._received_messages) == 1
    m = server._received_messages[0]
    assert isinstance(m, ClientMessage)
    assert m["client_id"] == 7 and m["train_samples"] == 10
    Server.receive_one(server, ClientToServer(client_id=8, train_samples=1, metrics={}))
    assert len(server._received_messages) == 2


def test_accept_parameters_with_typed_message():
    """The typed receive path feeds the legacy accept_parameters unchanged."""
    client = _fake_client()
    params = _params()
    Client.receive(client, ServerToClient(parameters=params))
    Client.accept_parameters(client)
    assert client._cached_parameters == params


def test_default_communicate_implementations_exist():
    """The base classes ship concrete (typed-API) default `communicate`s,
    so new algorithms only need to override them when they must;
    the abstract one remains only on the Node base.
    """
    assert "communicate" not in Server.__abstractmethods__
    assert "communicate" not in Client.__abstractmethods__
    assert "communicate" in Server.__mro__[1].__abstractmethods__  # Node


def test_model_state_in_typed_messages():
    """The named-parameter format flows through both directions."""
    state = {"w": torch.arange(6.0).reshape(2, 3), "b": torch.ones(2)}

    # server -> client: legacy dict carries "model_state" (and extras)
    d = ServerToClient(model_state=state, extras={"alpha": 0.1}).to_legacy_dict()
    assert d["model_state"]["w"].equal(state["w"])
    assert d["alpha"] == 0.1

    # client -> server: from_typed keeps the named state
    typed = ClientToServer(client_id=1, train_samples=10, metrics={}, model_state=state)
    legacy = ClientMessage.from_typed(typed)
    assert legacy["model_state"]["w"].equal(state["w"])

    # absent -> the key is absent
    assert "model_state" not in ServerToClient(parameters=_params()).to_legacy_dict()
    assert "model_state" not in ClientMessage.from_typed(ClientToServer(client_id=1, train_samples=2, metrics={}))


def test_accept_parameters_prefers_model_state():
    """`accept_parameters` handles the named format first, flat list second."""
    state = {"w": torch.arange(6.0).reshape(2, 3), "b": torch.ones(2)}
    client = _fake_client()
    client.set_model_state = lambda s: setattr(client, "_accepted_state", s)
    Client.receive(client, ServerToClient(model_state=state, parameters=_params()))
    Client.accept_parameters(client)
    assert client._accepted_state["w"].equal(state["w"])

    # only the flat list: the legacy path (set_parameters) applies
    params = _params()
    client2 = _fake_client()
    Client.receive(client2, ServerToClient(parameters=params))
    Client.accept_parameters(client2)
    assert client2._cached_parameters == params
