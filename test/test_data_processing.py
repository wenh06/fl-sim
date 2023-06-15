"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from fl_sim.data_processing import (  # noqa: F401
    # abstract base classes
    FedDataset,
    FedVisionDataset,
    FedNLPDataset,
    # datasets from FedML
    # FedCIFAR,  # the same as FedCIFAR currently
    FedCIFAR100,
    FedEMNIST,
    FedMNIST,
    FedRotatedCIFAR10,
    FedRotatedMNIST,
    FedShakespeare,
    FedSynthetic,
    # datasets from FedProx
    FedProxFEMNIST,
    FedProxMNIST,
    FedProxSent140,
    # libsvm datasets
    FedLibSVMDataset,
    libsvmread,
)  # noqa: F401


@torch.no_grad()
def test_FedCIFAR100():
    """ """
    ds = FedCIFAR100()
    assert ds.n_class == 100
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0

    ds = FedCIFAR100(transform=None)  # use default non-trivial transform
    train_dl, test_dl = ds.get_dataloader()
    assert len(train_dl) > 0 and len(test_dl) > 0


@torch.no_grad()
def test_FedEMNIST():
    ds = FedEMNIST()

    assert ds.n_class == 62
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0

    with pytest.warns(RuntimeWarning):
        ds = FedEMNIST(transform=None)


@torch.no_grad()
def test_FedMNIST():
    ds = FedMNIST()

    assert ds.n_class == 10
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0

    with pytest.warns(RuntimeWarning):
        ds = FedMNIST(transform=None)


@torch.no_grad()
def test_FedRotatedCIFAR10():
    ds = FedRotatedCIFAR10()
    ds = FedRotatedCIFAR10(num_rotations=4, transform=None)

    assert ds.n_class == 10

    ds.view_image(0, 0)

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0


@torch.no_grad()
def test_FedRotatedMNIST():
    ds = FedRotatedMNIST()
    ds = FedRotatedMNIST(num_rotations=2, transform=None)

    assert ds.n_class == 10

    ds.view_image(0, 0)

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0


@torch.no_grad()
def test_FedShakespeare():
    ds = FedShakespeare()

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    ds.view_sample(0)
    ds.view_sample(0, 0)

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 3
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0


@torch.no_grad()
def test_FedProxSent140():
    ds = FedProxSent140()

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    ds.view_sample(0, 0)

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0


@torch.no_grad()
def test_FedProxFEMNIST():
    """ """
    ds = FedProxFEMNIST()

    assert ds.n_class == 10
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0

    with pytest.warns(RuntimeWarning):
        ds = FedProxFEMNIST(transform=None)


@torch.no_grad()
def test_FedProxMNIST():
    """ """
    ds = FedProxMNIST()

    assert ds.n_class == 10
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)

    assert str(ds) == repr(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0

    with pytest.warns(RuntimeWarning):
        ds = FedProxMNIST(transform=None)


@torch.no_grad()
def test_FedSynthetic():
    """ """
    ds = FedSynthetic(1, 1, False, 30)

    assert repr(ds) == str(ds)

    train_dl, test_dl = ds.get_dataloader(client_idx=0)
    assert len(train_dl) > 0 and len(test_dl) > 0

    train_dl, test_dl = ds.get_dataloader(client_idx=None)
    assert len(train_dl) > 0 and len(test_dl) > 0

    candidate_models = ds.candidate_models
    assert len(candidate_models) > 0 and isinstance(candidate_models, dict)
    for model_name, model in candidate_models.items():
        assert isinstance(model_name, str) and isinstance(model, torch.nn.Module)

        model.eval()
        batch = next(iter(train_dl))
        loss = ds.criterion(model(batch[0]), batch[1])
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

        prob = model.predict_proba(batch[0][0], batched=False)
        pred = model.predict(batch[0][0], batched=False)
        assert isinstance(prob, np.ndarray) and prob.ndim == 1
        assert isinstance(pred, int)

        prob = model.predict_proba(batch[0], batched=True)
        pred = model.predict(batch[0], batched=True)
        assert isinstance(prob, np.ndarray) and prob.ndim == 2
        assert isinstance(pred, list) and len(pred) == batch[0].shape[0]
        eval_res = ds.evaluate(torch.from_numpy(prob), batch[1])
        assert isinstance(eval_res, dict) and len(eval_res) > 0
