"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

import torch
from torch_ecg.utils.misc import get_required_args

from fl_sim.models import ResNet10
from fl_sim.optimizers import available_optimizers, available_optimizers_plus, get_inner_solver, get_optimizer, get_oracle
from fl_sim.optimizers._register import _built_in_optimizers


def test_optimizers():
    local_model = ResNet10(10).train()
    dual_weights = [torch.randn_like(p) for p in local_model.parameters()]
    variance_buffer = [torch.randn_like(p) for p in local_model.parameters()]
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    config = {"lr": 1e-2}
    for optimizer_name in available_optimizers:
        model = ResNet10(10).train()
        optimizer = get_optimizer(optimizer_name, model.parameters(), config)
        assert isinstance(optimizer, torch.optim.Optimizer)
        required_kwargs = get_required_args(optimizer.step)
        step_kwargs = {}
        if "local_weights" in required_kwargs:
            step_kwargs["local_weights"] = local_model.parameters()
        if "dual_weights" in required_kwargs:
            step_kwargs["dual_weights"] = dual_weights
        if "variance_buffer" in required_kwargs:
            step_kwargs["variance_buffer"] = variance_buffer
        if "closure" in required_kwargs:

            def closure():
                optimizer.zero_grad()  # noqa: F821
                loss = criterion(model(x), y)  # noqa: F821
                loss.backward()
                return loss

            step_kwargs["closure"] = closure
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        try:
            optimizer.step(**step_kwargs)
        except Exception as e:
            print(step_kwargs)
            print(optimizer)
            import time

            time.sleep(3)
            raise e
        del optimizer, model

    for optimizer_name in available_optimizers_plus:
        model = ResNet10(10).train()
        if optimizer_name in available_optimizers:
            continue
        oracle = get_oracle(optimizer_name, model.parameters(), config)
        assert isinstance(oracle, torch.optim.Optimizer)
        required_kwargs = get_required_args(oracle.step)
        if "closure" in required_kwargs:

            def closure():
                oracle.zero_grad()  # noqa: F821
                loss = criterion(model(x), y)  # noqa: F821
                loss.backward()
                return loss

        else:
            closure = None
        oracle.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if closure is not None:
            oracle.step(closure)
        del oracle, model

    model = ResNet10(10).train()
    inner_solver = get_inner_solver(torch.optim.SGD, model.parameters(), config)
    assert isinstance(inner_solver, torch.optim.Optimizer)
    inner_solver.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    inner_solver.step()
    del inner_solver, model

    model = ResNet10(10).train()
    local_model = ResNet10(10).train()
    inner_solver = get_inner_solver("test-files/custom_optimizer.py", model.parameters(), config)
    assert isinstance(inner_solver, torch.optim.Optimizer)
    inner_solver.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    inner_solver.step(local_weights=local_model.parameters())
    del inner_solver, model, local_model

    _built_in_optimizers.pop("Custom")  # not CustomOptimizer

    model = ResNet10(10).train()
    local_model = ResNet10(10).train()
    inner_solver = get_inner_solver("test-files/custom_optimizer.CustomOptimizer", model.parameters(), config)
    assert isinstance(inner_solver, torch.optim.Optimizer)
    inner_solver.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    inner_solver.step(local_weights=local_model.parameters())
    del inner_solver, model, local_model

    model = ResNet10(10).train()
    innner_solver = get_inner_solver("SCAFFOLD", model.parameters(), config)
    assert isinstance(innner_solver, torch.optim.Optimizer)
    innner_solver.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    innner_solver.step(variance_buffer=variance_buffer)
    del innner_solver, model
