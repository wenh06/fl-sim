""" """

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parents[1].resolve()))

import numpy as np
import pytest
import torch

from fl_sim.data_processing.fed_synthetic import FedSynthetic
from fl_sim.nodes import Client, ClientConfig, ClientMessage, Server, ServerConfig


class DummySeverConfig(ServerConfig):
    # the (pydantic) configs support positional arguments,
    # mapped onto the fields in their declaration order
    def __init__(
        self,
        num_iters: int,
        num_clients: int,
    ) -> None:
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio=1.0,
        )


class DummyClientConfig(ClientConfig):
    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(
            batch_size,
            num_epochs,
            lr,
            optimizer="SGD",
        )


class DummyServer(Server):
    def _post_init(self) -> None:
        pass

    @property
    def client_cls(self) -> type:
        return DummyClient

    def communicate(self, target: "DummyClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        # do nothing
        pass

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": DummySeverConfig,
            "client": DummyClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return None


class DummyClient(Client):
    def communicate(self, target: "DummyServer"):
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "train_samples": len(self.train_loader.dataset),
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> None:
        # do nothing
        pass

    def train(self) -> None:
        # do nothing
        pass


def test_aggregate_results_from_json_log():
    json_log_file = list((Path(__file__).parents[1].resolve() / "test-files").glob("*.json"))[0]
    for part, metric in product(["train", "val"], ["acc", "loss"]):
        curve = Server.aggregate_results_from_json_log(json_log_file, part=part, metric=metric)
        assert len(curve) > 0


def test_get_norm():
    tensors = FedSynthetic(1, 1, False, 30).candidate_models["mlp_d1"].parameters()
    norm = Server.get_norm(tensors)
    assert isinstance(norm, float) and norm > 0
    tensors = None
    norm = Server.get_norm(tensors)
    assert np.isnan(norm)
    tensors = 1.2
    norm = Server.get_norm(tensors)
    assert norm == 1.2
    tensors = torch.randn(2, 3)
    norm = Server.get_norm(tensors)
    assert isinstance(norm, float) and norm > 0
    tensors = np.random.randn(2, 3)
    norm = Server.get_norm(tensors)
    assert isinstance(norm, float) and norm > 0
    tensors = [torch.randn(2, 3), np.random.randn(2, 3)]
    norm = Server.get_norm(tensors)
    assert isinstance(norm, float) and norm > 0
    tensors = torch.nn.Parameter(torch.randn(2, 3))
    norm = Server.get_norm(tensors)
    assert isinstance(norm, float) and norm > 0


def test_nodes():
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = DummySeverConfig(1, 10)
    client_config = DummyClientConfig(1, 1)
    s = DummyServer(model, dataset, server_config, client_config)
    assert str(s) == repr(s)

    # s.train_centralized()
    s.train("local")
    s.train("centralized")
    s.train()  # federated training by default

    # misc functionalities of Node classes
    data, labels = s.get_client_data(0)
    with pytest.raises(ValueError):
        s.get_client_data(len(s._clients))

    model = s.get_client_model(0)
    with pytest.raises(ValueError):
        s.get_client_model(len(s._clients))

    metrics = s.get_cached_metrics()
    metrics = s.get_cached_metrics(0)
    with pytest.raises(ValueError):
        s.get_cached_metrics(len(s._clients))

    assert str(s._clients[0]) == repr(s._clients[0])
    assert isinstance(s.is_convergent, bool)
    assert isinstance(s._clients[0].is_convergent, bool)

    del dataset, model, s


def _make_server(num_iters: int = 1):
    """A fresh (dummy) server with 10 clients on FedSynthetic."""
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    s = DummyServer(model, dataset, DummySeverConfig(num_iters, 10), DummyClientConfig(32, 1))
    return s


def test_node_gradient_utilities():
    s = _make_server()
    client = s._clients[0]

    # compute_gradients at the current parameters
    grads = client.compute_gradients()
    assert isinstance(grads, list) and len(grads) > 0
    assert all(isinstance(g, torch.Tensor) for g in grads)

    # compute_gradients at some other (given) parameters,
    # with the model state restored afterwards
    params = client.get_detached_model_parameters()
    for p in params:
        p += 0.01
    grads_at = client.compute_gradients(at=params)
    assert all(isinstance(g, torch.Tensor) for g in grads_at)

    # get_gradients, raw and normed
    raw = client.get_gradients()
    assert isinstance(raw, list)
    normed = client.get_gradients(norm="fro")
    assert isinstance(normed, float) and normed >= 0

    # set_parameters roundtrip
    new_params = [torch.randn_like(p) for p in params]
    client.set_parameters(new_params)
    for p_model, p_new in zip(client.model.parameters(), new_params):
        assert torch.allclose(p_model.data, p_new.data.to(client.device))

    del s


def test_server_sample_clients():
    s = _make_server()

    # by ratio (default of the dummy config: 1.0)
    selected = s._sample_clients()
    assert len(selected) == 10
    # by custom ratio
    assert len(s._sample_clients(clients_sample_ratio=0.3)) == 3
    # by number (overrides the ratio)
    assert len(s._sample_clients(clients_sample_num=5)) == 5
    # cap at the subset size
    assert len(s._sample_clients(clients_sample_num=100)) == 10
    # within a given subset
    selected = s._sample_clients(subset=[1, 2, 3], clients_sample_ratio=0.5)
    assert len(selected) == 2 and set(selected).issubset({1, 2, 3})

    del s


def test_server_parameter_aggregation():
    s = _make_server()
    client = s._clients[0]
    server_params = list(s.model.parameters())

    # start from a zeroed server model for exact assertions
    for p in server_params:
        p.data.zero_()

    ones_params = [torch.ones_like(p) for p in server_params]
    three_params = [torch.full_like(p, 3.0) for p in server_params]

    # add_parameters with ratio 1.0 makes the server model the given parameters
    s.add_parameters(ones_params, ratio=1.0)
    for sp in server_params:
        assert torch.allclose(sp.data, torch.ones_like(sp.data))

    # avg_parameters: equal-weight average of one message = the message parameters
    for p in server_params:
        p.data.zero_()
    s._received_messages = [
        ClientMessage(client_id=0, train_samples=100, metrics={}, parameters=ones_params),
    ]
    s.avg_parameters(size_aware=False)
    for sp in server_params:
        assert torch.allclose(sp.data, torch.ones_like(sp.data))

    # size_aware weighting: (1.0 * 300 + 3.0 * 100) / 400 = 1.5
    s._received_messages = [
        ClientMessage(client_id=0, train_samples=300, metrics={}, parameters=ones_params),
        ClientMessage(client_id=1, train_samples=100, metrics={}, parameters=three_params),
    ]
    s.avg_parameters(size_aware=True, inertia=0.0)
    for sp in server_params:
        assert torch.allclose(sp.data, torch.full_like(sp.data, 1.5))

    # inertia keeps half of the previous parameters: 0.5 * 1.5 + 0.5 * 3.0 = 2.25
    s._received_messages = [
        ClientMessage(client_id=0, train_samples=1, metrics={}, parameters=three_params),
    ]
    s.avg_parameters(size_aware=False, inertia=0.5)
    for sp in server_params:
        assert torch.allclose(sp.data, torch.full_like(sp.data, 2.25))

    with pytest.raises(AssertionError):
        s.avg_parameters(inertia=1.0)

    del s


def test_server_update_gradients():
    s = _make_server()
    grads = [torch.full_like(p, 0.5) for p in s.model.parameters()]
    s._received_messages = [
        ClientMessage(client_id=0, train_samples=100, metrics={}, gradients=grads),
        ClientMessage(client_id=1, train_samples=300, metrics={}, gradients=grads),
    ]
    s.update_gradients()
    for p in s.model.parameters():
        assert p.grad is not None and torch.allclose(p.grad, torch.full_like(p.grad, 0.5))

    # no messages: no-op without error
    s._received_messages = []
    s.update_gradients()

    del s


def test_server_aggregate_client_metrics():
    s = _make_server()

    with pytest.raises(ValueError, match="no metrics"):
        # a message without the "metrics" key (plain dict, as legacy senders may build)
        s._received_messages = [dict(client_id=0, train_samples=100)]
        s.aggregate_client_metrics()

    s._received_messages = [
        ClientMessage(client_id=0, train_samples=100, metrics={"train": {"acc": 0.6, "num_samples": 100}}),
        ClientMessage(client_id=1, train_samples=300, metrics={"train": {"acc": 0.8, "num_samples": 300}}),
    ]
    s.aggregate_client_metrics()
    # sample-weighted mean accuracy: (0.6*100 + 0.8*300) / 400 = 0.75
    assert s._metrics["train"]["acc"] == pytest.approx(0.75)
    assert s._metrics["train"]["num_samples"] == 400
    assert len(s._cached_metrics) > 0

    del s


def test_server_evaluate_centralized():
    s = _make_server()
    _, val_loader = s.dataset.get_dataloader()
    metrics = s.evaluate_centralized(val_loader)
    assert "num_samples" in metrics and metrics["num_samples"] > 0
    assert "acc" in metrics or "loss" in metrics
    del s


def test_server_misc_states():
    s = _make_server()

    # the (overridable) pre-iteration hook
    s.pre_iter()

    # typed communication: the default broadcast + upload round trip
    client = s._clients[0]
    s.communicate(client)
    assert "parameters" in client._received_messages
    client.communicate(s)
    assert len(s._received_messages) == 1

    # is_convergent reflects the status variable
    s._is_convergent = True
    assert s.is_convergent
    s._is_convergent = False

    # reset restores the initial state
    s._received_messages = [ClientMessage(client_id=0, train_samples=1, metrics={})]
    s._reset()
    assert s._received_messages == [] and s._metrics == {} and s._cached_metrics == []
    assert not s._complete_experiment and s._num_communications == 0

    del s


def test_server_four_phase_hooks():
    s = _make_server(num_iters=2)
    client = s._clients[0]

    # the eval gate: iteration 0 never evaluates
    s.n_iter = 0
    assert not s._should_evaluate()
    s.n_iter = 1  # (1 + 1) % 1 == 0 and n_iter > 0
    assert s._should_evaluate()

    # broadcast: downlink + accept
    s._broadcast(client)
    assert "parameters" in client._received_messages

    # local step with evaluation: metrics logged and cached on the client
    s._local_step(client, evaluate=True)
    assert len(s._received_messages) == 1
    assert len(client._cached_metrics) == 1

    # aggregate + global step
    s._aggregate_metrics(evaluate=True)
    s._global_step()
    assert s._received_messages == []

    # the no-op variant of the aggregate phase
    s._aggregate_metrics(evaluate=False)

    del s


class _PlainModel:
    """A deliberately NON-torch.nn.Module model: parameters are plain tensors."""

    def __init__(self) -> None:
        self.w = torch.zeros(3, 2)
        self.b = torch.zeros(3)


class NamedStateServer(DummyServer):
    """A server holding a non-module model, with the named exchange implemented."""

    def get_model_state(self):
        return {"w": self.model.w.detach().clone(), "b": self.model.b.detach().clone()}

    def set_model_state(self, state):
        self.model.w = state["w"].detach().clone().to(self.device)
        self.model.b = state["b"].detach().clone().to(self.device)


class NamedStateClient(DummyClient):
    """A client holding a non-module model, with the named exchange implemented."""

    def get_model_state(self):
        return {"w": self.model.w.detach().clone(), "b": self.model.b.detach().clone()}

    def set_model_state(self, state):
        self.model.w = state["w"].detach().clone().to(self.device)
        self.model.b = state["b"].detach().clone().to(self.device)


def test_node_model_state_default_module_path():
    """For nn.Module models the default methods wrap state_dict/load_state_dict."""
    s = _make_server()

    state = s.get_model_state()
    assert set(state.keys()) == set(s.model.state_dict().keys())
    for k, v in state.items():
        assert v.equal(s.model.state_dict()[k])
        assert not v.requires_grad  # detached clones

    # roundtrip: change the model, then restore from the saved state
    saved = s.get_model_state()
    with torch.no_grad():
        for p in s.model.parameters():
            p.add_(1.0)
    s.set_model_state(saved)
    for k, v in s.model.state_dict().items():
        assert v.equal(saved[k])

    # client side default too
    client = s._clients[0]
    cstate = client.get_model_state()
    client.set_model_state(cstate)
    for k, v in client.model.state_dict().items():
        assert v.equal(cstate[k])

    del s


def test_node_model_state_non_module_raises_by_default():
    s = _make_server()
    s.model = _PlainModel()  # not an nn.Module
    with pytest.raises(NotImplementedError, match="get_model_state"):
        s.get_model_state()
    with pytest.raises(NotImplementedError, match="set_model_state"):
        s.set_model_state({"w": torch.zeros(3, 2)})
    del s


def test_named_exchange_with_non_module_model():
    """End-to-end named exchange between nodes holding a non-module model."""
    dataset = FedSynthetic(1, 1, False, 30)
    model = _PlainModel()
    with torch.no_grad():
        model.w += 1.5
        model.b += 0.5
    server = NamedStateServer.__new__(NamedStateServer)
    # skip __init__: only the pieces the named exchange touches are needed
    server.model = model
    server.device = torch.device("cpu")
    server._received_messages = []

    client = NamedStateClient.__new__(NamedStateClient)
    client.model = _PlainModel()
    client.device = torch.device("cpu")
    client._received_messages = {}
    client._metrics = {}
    client._cached_metrics = []

    # typed downlink: ServerToClient(model_state=...) -> receive -> accept
    from fl_sim.messages import ServerToClient

    client.receive(ServerToClient(model_state=server.get_model_state()))
    assert "model_state" in client._received_messages
    client.accept_parameters()
    assert client.model.w.equal(model.w) and client.model.b.equal(model.b)

    # typed uplink: the client's named state reaches the server's legacy container
    from fl_sim.messages import ClientToServer

    client.communicate = lambda target: target.receive_one(
        ClientToServer(client_id=0, train_samples=1, metrics={}, model_state=client.get_model_state())
    )
    server._num_communications = 0
    client._communicate(server)
    assert server._num_communications == 1
    m = server._received_messages[0]
    assert m["model_state"]["w"].equal(model.w) and m["model_state"]["b"].equal(model.b)
