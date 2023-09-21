"""
"""

from ...data_processing.fed_synthetic import FedSynthetic
from ...data_processing.fedprox_femnist import FedProxFEMNIST
from ...utils.misc import experiment_indicator
from ._ifca import IFCAClientConfig, IFCAServer, IFCAServerConfig

__all__ = [
    "test_ifca",
]


@experiment_indicator("IFCA")
def test_ifca() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = IFCAServerConfig(3, 10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = IFCAClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = IFCAServer(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = IFCAServerConfig(4, 10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.2)
    client_config = IFCAClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = IFCAServer(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_ifca()
