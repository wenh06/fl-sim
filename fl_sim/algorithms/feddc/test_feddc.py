""" """

from ...data_processing.fed_synthetic import FedSynthetic
from ...utils.misc import experiment_indicator
from ._feddc import FedDCClientConfig, FedDCServer, FedDCServerConfig

__all__ = [
    "test_feddc",
]


@experiment_indicator("FedDC")
def test_feddc() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedDCServerConfig(
        10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7, alpha=0.01
    )
    client_config = FedDCClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedDCServer(model, dataset, server_config, client_config)
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedSynthetic, with a larger alpha")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedDCServerConfig(
        5, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 1.0, alpha=0.1
    )
    client_config = FedDCClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedDCServer(model, dataset, server_config, client_config)
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_feddc()
