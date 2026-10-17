""" """

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

import pytest
import torch

from fl_sim.nodes import ClientConfig, ServerConfig


def test_config_positional_mapping():
    # positional arguments map onto the fields in their declaration order
    sc = ServerConfig(10, 100, 0.5)
    assert sc.num_iters == 10
    assert sc.num_clients == 100
    assert sc.clients_sample_ratio == 0.5
    # defaults of the remaining fields
    assert sc.eval_every == 1 and sc.seed == 0 and sc.verbose == 1
    # the base configs derive an EMPTY algorithm name
    # (only conventionally-named subclasses derive a non-empty one)
    assert sc.algorithm == ""

    cc = ClientConfig(64, 5, 1e-3, "SGD")
    assert cc.batch_size == 64 and cc.num_epochs == 5 and cc.lr == 1e-3 and cc.optimizer == "SGD"
    assert cc.algorithm == ""


def test_config_string_first_positional_rejected():
    # the former leading `algorithm` positional argument has been removed
    with pytest.raises(TypeError, match="algorithm"):
        ServerConfig("FedAvg", 10, 100, 0.5)
    with pytest.raises(TypeError, match="algorithm"):
        ClientConfig("FedAvg", 64, 5, 1e-3, "SGD")


def test_config_too_many_positionals_rejected():
    # ServerConfig has 14 fields; the string guard fires only for a string
    # in the FIRST slot, so use non-strings to reach the arity check
    with pytest.raises(TypeError, match="positional"):
        ServerConfig(*([10, 100, 0.5] + [1] * 12))


def test_config_duplicate_positional_and_kwarg_rejected():
    # the third positional maps onto clients_sample_ratio, which is also a keyword
    with pytest.raises(TypeError, match="multiple values"):
        ServerConfig(10, 100, 0.5, clients_sample_ratio=0.4)


def test_config_algorithm_derivation_and_forcing():
    class FedProxServerConfig(ServerConfig):
        __name__ = "FedProxServerConfig"

    # derived from the class name when absent
    cfg = FedProxServerConfig(10, 100, 0.5)
    assert cfg.algorithm == "FedProx"

    # explicitly passed with a different value: warn and force
    with pytest.warns(RuntimeWarning, match="FedProx"):
        cfg = FedProxServerConfig(10, 100, 0.5, algorithm="FedAvg")
    assert cfg.algorithm == "FedProx"

    # the base classes derive an empty name, hence keep whatever passed
    cfg = ServerConfig(10, 100, 0.5, algorithm="FedAvg")
    assert cfg.algorithm == "FedAvg"


def test_config_scheduler_normalization():
    assert ClientConfig(64, 5, 1e-3, "SGD").scheduler == {"name": "none"}
    cc = ClientConfig(64, 5, 1e-3, "SGD", scheduler={"name": "none"})
    assert cc.scheduler == {"name": "none"}


def test_config_gpus_filtering():
    num_gpus = torch.cuda.device_count()
    default = ServerConfig(10, 100, 0.5)
    assert default.visiable_gpus == list(range(num_gpus))

    if num_gpus == 0:
        with pytest.warns(UserWarning, match="not available"):
            cfg = ServerConfig(10, 100, 0.5, visiable_gpus=[0, 1])
        assert cfg.visiable_gpus == []
    else:
        cfg = ServerConfig(10, 100, 0.5, visiable_gpus=[0])
        assert cfg.visiable_gpus == [0]
        with pytest.warns(UserWarning, match="not available"):
            cfg = ServerConfig(10, 100, 0.5, visiable_gpus=[num_gpus])
        assert cfg.visiable_gpus == []


def test_config_extra_kwargs_become_attributes():
    sc = ServerConfig(10, 100, 0.5, custom_attribute=42)
    assert sc.custom_attribute == 42
    cc = ClientConfig(64, 5, 1e-3, "SGD", custom_attribute="foo")
    assert cc.custom_attribute == "foo"


def test_config_extra_repr_keys_empty():
    assert ServerConfig(10, 100, 0.5).extra_repr_keys() == []
    assert ClientConfig(64, 5, 1e-3, "SGD").extra_repr_keys() == []
