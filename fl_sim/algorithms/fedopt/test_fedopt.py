"""
"""

from ...utils.misc import experiment_indicator
from ...data_processing.fed_synthetic import FedSynthetic
from ...data_processing.fedprox_femnist import FedProxFEMNIST

from ._fedopt import (
    FedAvgServer,
    FedAvgServerConfig,
    FedAvgClientConfig,
    FedAdagradServer,
    FedAdagradServerConfig,
    FedAdagradClientConfig,
    FedYogiServer,
    FedYogiServerConfig,
    FedYogiClientConfig,
    FedAdamServer,
    FedAdamServerConfig,
    FedAdamClientConfig,
)


__all__ = [
    "test_fedopt",
    "test_fedavg",
    "test_fedadagrad",
    "test_fedyogi",
    "test_fedadam",
]


@experiment_indicator("FedOpt")
def test_fedopt(algorithm: str) -> None:
    """ """
    assert algorithm.lower() in [
        "avg",
        "adagrad",
        "yogi",
        "adam",
    ]
    print(f"Using algorithm {algorithm}")
    if algorithm.lower() == "avg":
        client_config_cls = FedAvgClientConfig
        server_config_cls = FedAvgServerConfig
        server_cls = FedAvgServer
    elif algorithm.lower() == "adagrad":
        client_config_cls = FedAdagradClientConfig
        server_config_cls = FedAdagradServerConfig
        server_cls = FedAdagradServer
    elif algorithm.lower() == "yogi":
        client_config_cls = FedYogiClientConfig
        server_config_cls = FedYogiServerConfig
        server_cls = FedYogiServer
    elif algorithm.lower() == "adam":
        client_config_cls = FedAdamClientConfig
        server_config_cls = FedAdamServerConfig
        server_cls = FedAdamServer

    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = server_config_cls(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = client_config_cls(dataset.DEFAULT_BATCH_SIZE, 20)
    s = server_cls(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = server_config_cls(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.2)
    client_config = client_config_cls(dataset.DEFAULT_BATCH_SIZE, 20)
    s = server_cls(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s


def test_fedavg() -> None:
    test_fedopt("avg")


def test_fedadagrad() -> None:
    test_fedopt("adagrad")


def test_fedyogi() -> None:
    test_fedopt("yogi")


def test_fedadam() -> None:
    test_fedopt("adam")


if __name__ == "__main__":
    test_fedopt("avg")
    test_fedopt("adagrad")
    test_fedopt("yogi")
    test_fedopt("adam")
