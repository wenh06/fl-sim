"""
"""

import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

import numpy as np
import pytest
import torch
from torch_ecg.utils import get_kwargs

from fl_sim.models import CNNFEMnist
from fl_sim.optimizers import get_optimizer
from fl_sim.utils._download_data import url_is_reachable
from fl_sim.utils.misc import (
    add_kwargs,
    default_dict_to_dict,
    experiment_indicator,
    find_longest_common_substring,
    get_scheduler,
    get_scheduler_info,
    make_serializable,
    ordered_dict_to_dict,
)
from fl_sim.utils.torch_compat import torch_norm


def test_get_scheduler():
    num_classes = 62
    init_lr = 0.05

    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("none", optimizer, None)
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr == init_lr
    del X, y, loss, model, optimizer, scheduler

    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("cosine", optimizer, dict(T_max=10))
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr != init_lr
    del X, y, loss, model, optimizer, scheduler

    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("step", optimizer, dict(step_size=1))
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr != init_lr
    del X, y, loss, model, optimizer, scheduler

    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("multi_step", optimizer, dict(milestones=[1]))
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr != init_lr
    del X, y, loss, model, optimizer, scheduler

    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("exponential", optimizer, dict(gamma=0.1))
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr != init_lr
    del X, y, loss, model, optimizer, scheduler

    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("cyclic", optimizer, dict(base_lr=0.01, max_lr=0.1))
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr != init_lr
    del X, y, loss, model, optimizer, scheduler

    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("one_cycle", optimizer, dict(max_lr=0.1, epochs=10, steps_per_epoch=1))
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr != init_lr
    del X, y, loss, model, optimizer, scheduler

    patience = 1
    model = CNNFEMnist(num_classes).train()
    optimizer = get_optimizer("SGD", model.parameters(), dict(lr=init_lr))
    scheduler = get_scheduler("reduce_on_plateau", optimizer, dict(mode="min", patience=patience))
    X = torch.randn(12, 1, 28, 28)
    y = torch.randint(0, num_classes, (12,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    for idx in range(patience + 2):
        scheduler.step(0.1)
    current_lr = scheduler._last_lr[0]
    assert current_lr < init_lr
    del X, y, loss, model, optimizer, scheduler


def test_get_scheduler_info():
    for name in [
        "step",
        "multi_step",
        "exponential",
        "cosine",
        "cyclic",
        "one_cycle",
        "reduce_on_plateau",
    ]:
        info = get_scheduler_info(name)
        assert isinstance(info, dict) and list(info) == [
            "class",
            "required_args",
            "optional_args",
        ]


def test_add_kwargs():
    def func(a, b=1):
        return a + b

    new_func = add_kwargs(func, xxx="yyy", zzz=None)

    assert new_func(2) == new_func(2, xxx="a", zzz=100) == 3
    assert get_kwargs(new_func) == {"b": 1, "xxx": "yyy", "zzz": None}

    class Dummy:
        def func(self, a, b=1):
            return a + b

    dummy = Dummy()
    new_func = add_kwargs(dummy.func, xxx="yyy", zzz=None)

    assert new_func(2) == new_func(2, xxx="a", zzz=100) == 3
    assert get_kwargs(new_func) == {"b": 1, "xxx": "yyy", "zzz": None}


def test_make_serializable():
    x = np.array([1, 2, 3])
    assert make_serializable(x) == [1, 2, 3]
    x = {"a": np.array([1, 2, 3]), "b": [np.array([4, 5, 6]), np.array([7, 8, 9])]}
    assert make_serializable(x) == {"a": [1, 2, 3], "b": [[4, 5, 6], [7, 8, 9]]}
    x = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    assert make_serializable(x) == [[1, 2, 3], [4, 5, 6]]
    x = (np.array([1, 2, 3]), np.array([4, 5, 6]).mean())
    obj = make_serializable(x)
    assert obj == [[1, 2, 3], 5.0]
    assert isinstance(obj[1], float) and isinstance(x[1], np.float64)


def test_ordered_dict_to_dict():
    obj = OrderedDict([("a", 1), ("b", [2, 3, OrderedDict([("xx", 1)])]), ("c", {"d": 4})])
    assert ordered_dict_to_dict(obj) == {"a": 1, "b": [2, 3, {"xx": 1}], "c": {"d": 4}}


def test_default_dict_to_dict():
    obj = defaultdict(lambda: defaultdict(list))
    obj["a"]["b"].append(1)
    obj["a"]["b"].append(dict(c=2))
    new_obj = defaultdict(dict)
    new_obj["xx"]["yy"] = [1, 2, 3]
    obj["i"]["j"] = new_obj
    assert default_dict_to_dict(obj) == {
        "a": {"b": [1, {"c": 2}]},
        "i": {"j": {"xx": {"yy": [1, 2, 3]}}},
    }


def test_find_longest_common_substring():
    assert find_longest_common_substring(["abc", "ab", "abcd"]) == "ab"
    assert find_longest_common_substring(["abc", "ab", "abcd"], min_len=3) == ""
    assert find_longest_common_substring(["abxxxc", "abxxx", "abxxxcd"], ignore="xxx") == "ab"


@experiment_indicator("Dummy")
def test_experiment_indicator():
    pass


def test_url_is_reachable():
    assert url_is_reachable("https://www.cloudflare.com/")
    assert not url_is_reachable("https://www.google.com/xxx")


def test_torch_norm():
    a = torch.arange(9, dtype=torch.float) - 4
    b = a.reshape((3, 3))
    b_sp = b.clone().to_sparse()
    assert torch.allclose(torch_norm(a, dtype=torch.float64), torch.tensor(7.7460, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(torch_norm(b, dtype=torch.float64), torch.tensor(7.7460, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(torch_norm(a, float("inf"), dtype=torch.float64), torch.tensor(4.0, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(torch_norm(b, float("inf"), dtype=torch.float64), torch.tensor(4.0, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(torch_norm(b, p="nuc", dtype=torch.float64), torch.tensor(9.7980, dtype=torch.float64), atol=1e-4)
    b_sp = b.clone().to_sparse()
    assert torch.allclose(torch_norm(b_sp, float("inf")), torch.tensor(4.0), atol=1e-4)
    with pytest.raises(ValueError):
        torch_norm(b_sp, dtype=torch.float64)
    with pytest.raises(ValueError):
        torch_norm(b_sp, p="nuc", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch_norm(b_sp, p="xxx")
    with pytest.raises(RuntimeError):
        torch_norm(a, p="xxx")

    c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
    assert torch.allclose(
        torch_norm(c, dim=0, dtype=torch.float64), torch.tensor([1.4142, 2.2361, 5.0000], dtype=torch.float64), atol=1e-4
    )
    assert torch.allclose(
        torch_norm(c, dim=1, dtype=torch.float64), torch.tensor([3.7417, 4.2426], dtype=torch.float64), atol=1e-4
    )
    assert torch.allclose(
        torch_norm(c, p=1, dim=1, dtype=torch.float64), torch.tensor([6.0, 6.0], dtype=torch.float64), atol=1e-4
    )
    c_sp = c.clone().to_sparse()
    assert torch.allclose(torch_norm(c_sp, p=1), torch.tensor(12.0), atol=1e-4)
    with pytest.raises(RuntimeError):
        torch_norm(c_sp, p=1, dim=1)
    with pytest.raises(ValueError):
        torch_norm(c_sp, p="fro", dtype=torch.float64)
    with pytest.raises(ValueError):
        torch_norm(c_sp, p="nuc", dtype=torch.float64)

    d = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
    assert torch.allclose(
        torch_norm(d, dim=(1, 2), dtype=torch.float64), torch.tensor([3.7417, 11.2250], dtype=torch.float64), atol=1e-4
    )
    assert torch.allclose(torch_norm(d[0, :, :], dtype=torch.float64), torch.tensor(3.7417, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(torch_norm(d[1, :, :], dtype=torch.float64), torch.tensor(11.2250, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(
        torch_norm(d, p="nuc", dim=[1, 2], dtype=torch.float64), torch.tensor([4.2426, 11.4018], dtype=torch.float64), atol=1e-4
    )
    d_sp = d.clone().to_sparse()
    assert torch.allclose(torch_norm(d_sp), torch.tensor(11.8322), atol=1e-4)
    with pytest.raises(RuntimeError):
        torch_norm(d_sp, p="xxx")
