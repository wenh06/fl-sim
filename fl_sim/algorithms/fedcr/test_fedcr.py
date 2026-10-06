""" """

from ...data_processing.fed_synthetic import FedSynthetic
from ...utils.misc import experiment_indicator
from ._fedcr import FedCRClientConfig, FedCRServer, FedCRServerConfig

__all__ = [
    "test_fedcr",
]


@experiment_indicator("FedCR")
def test_fedcr() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedCRServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7, beta=0.005)
    client_config = FedCRClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedCRServer(model, dataset, server_config, client_config)
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedSynthetic, without the CMI regularizer (beta = 0)")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedCRServerConfig(5, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 1.0, beta=0.0)
    client_config = FedCRClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedCRServer(model, dataset, server_config, client_config)
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_fedcr()
