""" """

from ...data_processing.fed_synthetic import FedSynthetic
from ...utils.misc import experiment_indicator
from ._fpfc import FPFCClientConfig, FPFCServer, FPFCServerConfig

__all__ = [
    "test_fpfc",
]


@experiment_indicator("FPFC")
def test_fpfc() -> None:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FPFCServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7, rho=1.0, xi=1e-4, lam=0.6, a=3.7)
    client_config = FPFCClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FPFCServer(model, dataset, server_config, client_config)
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedSynthetic, with a larger lam")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FPFCServerConfig(5, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 1.0, rho=1.0, xi=1e-4, lam=4.0, a=3.7)
    client_config = FPFCClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FPFCServer(model, dataset, server_config, client_config)
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_fpfc()
