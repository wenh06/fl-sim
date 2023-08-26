"""
"""

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parents[1].resolve()))

import numpy as np
import pytest
import torch

from fl_sim.nodes import (
    Server,
    Client,
    ServerConfig,
    ClientConfig,
    ClientMessage,
)
from fl_sim.data_processing.fed_synthetic import FedSynthetic


class DummySeverConfig(ServerConfig):
    def __init__(
        self,
        num_iters: int,
        num_clients: int,
    ) -> None:
        super().__init__(
            "Dummy",
            num_iters,
            num_clients,
            clients_sample_ratio=1.0,
            optimizer="SGD",
            lr=1e-2,
        )


class DummyClientConfig(ClientConfig):
    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(
            "Dummy",
            "SGD",
            batch_size,
            num_epochs,
            lr,
        )


class DummyServer(Server):
    def _post_init(self) -> None:
        pass

    @property
    def client_cls(self) -> type:
        return DummyClient

    @property
    def required_config_fields(self) -> List[str]:
        return ["num_iters", "num_clients"]

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
    @property
    def required_config_fields(self) -> List[str]:
        return ["batch_size", "num_epochs", "lr"]

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
    json_log_file = list(
        (Path(__file__).parents[1].resolve() / "test-files").glob("*.json")
    )[0]
    for part, metric in product(["train", "val"], ["acc", "loss"]):
        curve = Server.aggregate_results_from_json_log(
            json_log_file, part=part, metric=metric
        )
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
