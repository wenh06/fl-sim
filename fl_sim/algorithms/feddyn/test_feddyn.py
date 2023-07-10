"""
"""

from ...utils.misc import experiment_indicator
from ...data_processing.fed_synthetic import FedSynthetic
from ...data_processing.fedprox_femnist import FedProxFEMNIST

from ._feddyn import FedDynServerConfig, FedDynClientConfig, FedDynServer


__all__ = [
    "test_feddyn",
]


@experiment_indicator("FedDyn")
def test_feddyn() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedDynServerConfig(
        10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, p=0.2, stochastic=False, vr=True
    )
    client_config = FedDynClientConfig(dataset.DEFAULT_BATCH_SIZE, 30, vr=True)
    s = FedDynServer(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = FedDynServerConfig(
        10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, p=0.2, stochastic=True, vr=True
    )
    client_config = FedDynClientConfig(dataset.DEFAULT_BATCH_SIZE, 30, vr=True)
    s = FedDynServer(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_feddyn()
