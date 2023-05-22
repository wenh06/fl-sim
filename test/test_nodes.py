"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

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
    ):
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
    ):
        super().__init__(
            "Dummy",
            "SGD",
            batch_size,
            num_epochs,
            lr,
        )


class DummyServer(Server):
    def _post_init(self):
        pass

    @property
    def client_cls(self):
        return DummyClient

    @property
    def required_config_fields(self):
        return ["num_iters", "num_clients"]

    def communicate(self, target: "DummyClient"):
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
    def doi(self):
        return None


class DummyClient(Client):
    @property
    def required_config_fields(self):
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


def test_nodes():
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = DummySeverConfig(1, 10)
    client_config = DummyClientConfig(1, 1)
    s = DummyServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_nodes()
    print("Test nodes succeeded!")
