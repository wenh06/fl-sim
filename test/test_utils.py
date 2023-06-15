"""
"""

import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

import numpy as np
import torch
from torch_ecg.utils import get_kwargs

from fl_sim.models import CNNFEMnist
from fl_sim.optimizers import get_optimizer
from fl_sim.utils.misc import (
    get_scheduler,
    add_kwargs,
    ndarray_to_list,
    ordered_dict_to_dict,
    default_dict_to_dict,
)


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
    scheduler = get_scheduler(
        "one_cycle", optimizer, dict(max_lr=0.1, epochs=10, steps_per_epoch=1)
    )
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
    scheduler = get_scheduler(
        "reduce_on_plateau", optimizer, dict(mode="min", patience=patience)
    )
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


def test_ndarray_to_list():
    obj = np.array([1, 2, 3])
    assert ndarray_to_list(obj) == [1, 2, 3]
    obj = {"a": np.array([1, 2, 3]), "b": [np.array([4, 5, 6]), np.array([7, 8, 9])]}
    assert ndarray_to_list(obj) == {"a": [1, 2, 3], "b": [[4, 5, 6], [7, 8, 9]]}


def test_ordered_dict_to_dict():
    obj = OrderedDict(
        [("a", 1), ("b", [2, 3, OrderedDict([("xx", 1)])]), ("c", {"d": 4})]
    )
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


if __name__ == "__main__":
    test_get_scheduler()
    test_add_kwargs()
    test_ndarray_to_list()
    test_ordered_dict_to_dict()
    test_default_dict_to_dict()
