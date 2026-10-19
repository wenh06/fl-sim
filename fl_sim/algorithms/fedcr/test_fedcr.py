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
    # the named exchange carries the feature extractor only;
    # the personal classifier head never enters the model_state
    head = s._head_name
    encoder_state = s._encoder_state(model)
    assert all(not (k == head or k.startswith(head + ".")) for k in encoder_state)
    assert set(encoder_state) | {n for n, _ in model.named_parameters() if n == head or n.startswith(head + ".")} == {
        n for n, _ in model.named_parameters()
    }
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
