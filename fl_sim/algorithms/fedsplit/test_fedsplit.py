"""
"""

from ...utils.misc import experiment_indicator
from ...data_processing.fed_synthetic import FedSynthetic
from ...data_processing.fedprox_femnist import FedProxFEMNIST

from ._fedsplit import (
    FedSplitServer,
    FedSplitClientConfig,
    FedSplitServerConfig,
)


__all__ = [
    "test_fedsplit",
]


@experiment_indicator("FedSplit")
def test_fedsplit() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedSplitServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = FedSplitClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedSplitServer(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = FedSplitServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.2)
    client_config = FedSplitClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedSplitServer(model, dataset, server_config, client_config)
    # s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_fedsplit()
