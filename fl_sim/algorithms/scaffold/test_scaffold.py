"""
"""

from ...utils.misc import experiment_indicator
from ...data_processing.fed_synthetic import FedSynthetic
from ...data_processing.fedprox_femnist import FedProxFEMNIST

from ._scaffold import SCAFFOLDServerConfig, SCAFFOLDClientConfig, SCAFFOLDServer


__all__ = [
    "test_scaffold",
]


@experiment_indicator("SCAFFOLD")
def test_scaffold() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = SCAFFOLDServerConfig(
        10,
        dataset.DEFAULT_TRAIN_CLIENTS_NUM,
        0.5,
        lr=0.01,
    )
    client_config = SCAFFOLDClientConfig(
        dataset.DEFAULT_BATCH_SIZE,
        30,
        lr=0.005,
        control_variate_update_rule=1,
    )
    s = SCAFFOLDServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = SCAFFOLDServerConfig(
        10,
        dataset.DEFAULT_TRAIN_CLIENTS_NUM,
        0.5,
        lr=0.01,
    )
    client_config = SCAFFOLDClientConfig(
        dataset.DEFAULT_BATCH_SIZE,
        30,
        lr=0.005,
        control_variate_update_rule=2,
    )
    s = SCAFFOLDServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_scaffold()
