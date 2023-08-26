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
from fl_sim.data_processing.fed_dataset import NLPDataset


def test_FedVisionDataset():
    from PIL import Image

    for shape in [(32, 32), (32, 32, 1), (32, 32, 3), (1, 32, 32), (3, 32, 32)]:
        img = FedVisionDataset.show_image(torch.randn(*shape))
        assert isinstance(img, Image.Image)
        img = FedVisionDataset.show_image(np.random.randn(*shape))
        assert isinstance(img, Image.Image)


@torch.no_grad()
def test_FedCIFAR100():
    """ """
    ds = FedCIFAR100()
    assert ds.n_class == 100
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    assert isinstance(ds.get_classes(torch.tensor([0, 1])), list)
    assert isinstance(ds.get_class(torch.tensor(0)), str)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

    ds.view_image(0, 0)
    ds.random_grid_view(3, 3, save_path="test_FedCIFAR100.pdf")

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

    ds.load_partition_data_distributed(0)
    ds.load_partition_data_distributed(1)
    ds.load_partition_data()


@torch.no_grad()
def test_FedEMNIST():
    ds = FedEMNIST()

    assert str(ds) == repr(ds)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

    assert ds.n_class == 62
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)
    ds.random_grid_view(3, 3, save_path="test_FedEMNIST.pdf")

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

    assert str(ds) == repr(ds)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

    assert ds.n_class == 10
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)
    ds.random_grid_view(3, 3, save_path="test_FedMNIST.pdf")

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

    assert str(ds) == repr(ds)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)
    assert ds.n_class == 10

    ds.view_image(0, 0)
    ds.random_grid_view(3, 3, save_path="test_FedRotatedCIFAR10.pdf")

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

    assert str(ds) == repr(ds)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)
    assert ds.n_class == 10

    ds.view_image(0, 0)
    ds.random_grid_view(3, 3, save_path="test_FedRotatedMNIST.pdf")

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
    assert isinstance(ds.get_word_dict(), dict)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

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

        pred = model.pipeline(
            "Yonder comes my master, your brother.",
            char_to_id=ds.char_to_id,
            id_to_char=ds.id_to_word,
        )
        assert isinstance(pred, str)

    ds.load_partition_data_distributed(0)
    ds.load_partition_data_distributed(1)
    ds.load_partition_data()


@torch.no_grad()
def test_FedProxSent140():
    ds = FedProxSent140()

    assert str(ds) == repr(ds)
    assert isinstance(ds.get_word_dict(), dict)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

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

    assert str(ds) == repr(ds)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

    assert ds.n_class == 10
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)
    ds.random_grid_view(3, 3, save_path="test_FedProxFEMNIST.pdf")

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

    assert str(ds) == repr(ds)
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

    assert ds.n_class == 10
    assert len(ds._client_ids_train) == ds.DEFAULT_TRAIN_CLIENTS_NUM
    assert len(ds._client_ids_test) == ds.DEFAULT_TEST_CLIENTS_NUM

    ds.view_image(0, 0)
    ds.random_grid_view(3, 3, save_path="test_FedProxMNIST.pdf")

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
    assert isinstance(ds.doi, list) and all(isinstance(d, str) for d in ds.doi)

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

    ds.load_partition_data_distributed(0)
    ds.load_partition_data_distributed(1)
    ds.load_partition_data()


def test_NLPDataset():
    from collections import OrderedDict

    ds = NLPDataset(
        dataset=[
            ("Not good not bad.", 2),
            ("It is funny.", 1),
            ("Not interesting at all!", 0),
        ],
        input_columns=["sentence"],
        label_names=["negative", "positive", "neutral"],
        shuffle=True,
    )

    assert ds.dataset_name is None
    assert str(ds) == repr(ds)

    assert len(ds) == 3
    assert isinstance(ds[0], tuple) and len(ds[0]) == 2
    assert isinstance(ds[0:2], list) and len(ds[0:2]) == 2

    assert ds._format_as_dict(("It is funny", 1)) == (
        OrderedDict([("sentence", "It is funny")]),
        1,
    )

    ds.filter_by_labels_(labels_to_keep=[0, 1])
    assert len(ds) == 2

    assert (
        NLPDataset._gen_input({"sentence": "It is funny."}, ["sentence"])
        == "It is funny."
    )

    for schema in [
        ("premise", "hypothesis", "label"),
        ("question", "sentence", "label"),
        ("sentence1", "sentence2", "label"),
        ("question1", "question2", "label"),
        ("question", "sentence", "label"),
        ("text", "label"),
        ("sentence", "label"),
        ("document", "summary"),
        ("content", "summary"),
        ("label", "review"),
    ]:
        input_columns, output_column = NLPDataset._split_dataset_columns(schema)
        assert isinstance(input_columns, tuple) and isinstance(output_column, str)
        assert all(isinstance(col, str) for col in input_columns)

    ds = NLPDataset.from_huggingface_dataset("sst2", split="train")
    assert len(ds) > 0
