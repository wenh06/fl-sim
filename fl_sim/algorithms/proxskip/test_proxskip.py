"""
"""

from ...utils.misc import experiment_indicator
from ...data_processing.fed_synthetic import FedSynthetic
from ...data_processing.fedprox_femnist import FedProxFEMNIST

from ._proxskip import ProxSkipServerConfig, ProxSkipClientConfig, ProxSkipServer


__all__ = [
    "test_proxskip",
]


@experiment_indicator("ProxSkip")
def test_proxskip() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = ProxSkipServerConfig(
        10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, p=0.2, vr=False
    )
    client_config = ProxSkipClientConfig(
        dataset.DEFAULT_BATCH_SIZE, 30, lr=0.005, vr=False
    )
    s = ProxSkipServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = ProxSkipServerConfig(
        10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, p=0.2, vr=True
    )
    client_config = ProxSkipClientConfig(
        dataset.DEFAULT_BATCH_SIZE, 30, lr=0.005, vr=True
    )
    s = ProxSkipServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_proxskip()
