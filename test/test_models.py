"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

import numpy as np
import torch

from fl_sim.models import (
    MLP,
    FedPDMLP,
    CNNMnist,
    CNNFEMnist,
    CNNFEMnist_Tiny,
    CNNCifar,
    CNNCifar_Small,
    CNNCifar_Tiny,
    RNN_OriginalFedAvg,
    RNN_StackOverFlow,
    RNN_Sent140,
    ResNet18,
    ResNet10,
    LogisticRegression,
    SVC,
    SVR,
    reset_parameters,
)
from fl_sim.data_processing import FedShakespeare


@torch.no_grad()
def test_models():
    """ """
    model = CNNFEMnist_Tiny().eval()
    inp = torch.rand(2, 1, 28, 28)
    out = model(inp)
    assert out.shape == (2, 62)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 62)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    reset_parameters(model)

    another_model = CNNFEMnist_Tiny()
    for norm in ["inf", "fro", float("inf"), -np.inf, 1, 2]:
        assert isinstance(model.diff(another_model, norm=norm), float)
    model_raw_diff = model.diff(another_model, norm=None)
    assert isinstance(model_raw_diff, list) and all(
        isinstance(p, torch.Tensor) for p in model_raw_diff
    )

    model = CNNFEMnist_Tiny(num_classes=10).eval()
    inp = torch.rand(2, 1, 28, 28)
    out = model(inp)
    assert out.shape == (2, 10)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 10)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    model = CNNFEMnist().eval()
    inp = torch.rand(2, 1, 28, 28)
    out = model(inp)
    assert out.shape == (2, 62)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 62)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    model = CNNFEMnist(num_classes=10).eval()
    inp = torch.rand(2, 1, 28, 28)
    out = model(inp)
    assert out.shape == (2, 10)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 10)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    model = CNNCifar(num_classes=10).eval()
    inp = torch.rand(2, 3, 32, 32)
    out = model(inp)
    assert out.shape == (2, 10)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 10)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    model = CNNCifar_Small(num_classes=100).eval()
    inp = torch.rand(2, 3, 32, 32)
    out = model(inp)
    assert out.shape == (2, 100)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 100)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    model = CNNCifar_Tiny(num_classes=100).eval()
    inp = torch.rand(2, 3, 32, 32)
    out = model(inp)
    assert out.shape == (2, 100)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 100)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    for num_classes in [10, 62]:
        model = CNNMnist(num_classes=num_classes).eval()
        inp = torch.rand(2, 1, 28, 28)
        out = model(inp)
        assert out.shape == (2, num_classes)
        prob = model.predict_proba(inp, batched=True)
        assert prob.shape == (2, num_classes)
        pred = model.predict(inp, batched=True)
        assert len(pred) == 2

    model = MLP(dim_in=28 * 28, dim_hidden=100, dim_out=62).eval()
    inp = torch.rand(2, 1, 28, 28)
    out = model(inp)
    assert out.shape == (2, 62)
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 62)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2

    model = MLP(dim_in=50, dim_hidden=100, dim_out=12, ndim=1).eval()
    inp = torch.rand(2, 5, 10)
    out = model(inp)
    assert out.shape == (2, 12)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 12)

    model = MLP(dim_in=50, dim_hidden=100, dim_out=12, ndim=0).eval()
    inp = torch.rand(2, 50)
    out = model(inp)
    assert out.shape == (2, 12)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 12)

    model = FedPDMLP(dim_in=28 * 28, dim_hidden=100, dim_out=62).eval()
    inp = torch.rand(2, 1, 28, 28)
    out = model(inp)
    assert out.shape == (2, 62)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 62)

    model = FedPDMLP(dim_in=50, dim_hidden=100, dim_out=12, ndim=1).eval()
    inp = torch.rand(2, 5, 10)
    out = model(inp)
    assert out.shape == (2, 12)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 12)

    model = FedPDMLP(dim_in=50, dim_hidden=100, dim_out=12, ndim=0).eval()
    inp = torch.rand(2, 50)
    out = model(inp)
    assert out.shape == (2, 12)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 12)

    model = ResNet18(num_classes=10).eval()
    inp = torch.rand(2, 3, 32, 32)
    out = model(inp)
    assert out.shape == (2, 10)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 10)

    model = ResNet10(num_classes=10).eval()
    inp = torch.rand(2, 3, 32, 32)
    out = model(inp)
    assert out.shape == (2, 10)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 10)

    model = RNN_OriginalFedAvg(embedding_dim=8, vocab_size=90, hidden_size=256).eval()
    inp = torch.randint(0, 90, (2, 50))
    out = model(inp)
    assert out.shape == (2, 90, 50)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2 and all([len(p) == 90 for p in pred])
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 90, 50)
    pred = model.pipeline(
        "Yonder comes my master, your brother.", char_to_id=FedShakespeare.word_dict
    )

    model = RNN_StackOverFlow().eval()
    inp = torch.randint(0, 1000, (2, 50))
    out = model(inp)
    assert out.shape == (2, 10004, 50)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2 and all([len(p) == 10004 for p in pred])
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 10004, 50)
    # TODO: add test for pipeline

    model = LogisticRegression(num_features=50, num_classes=12).eval()
    inp = torch.rand(2, 50)
    out = model(inp)
    assert out.shape == (2, 12)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 12)

    model = SVC(num_features=50, num_classes=12).eval()
    inp = torch.rand(2, 50)
    out = model(inp)
    assert out.shape == (2, 12)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 12)

    model = SVR(num_features=50).eval()
    inp = torch.rand(2, 50)
    out = model(inp)
    assert out.shape == (2, 1)
    pred = model.predict(inp)
    assert pred.shape == (2, 1)
    pred = model.predict(inp[0])
    assert pred.shape == (1,)

    model = RNN_Sent140().eval()
    inp = torch.randint(0, model.word_embeddings.vocab_size, (2, 50))
    out = model(inp)
    assert out.shape == (2, 2)
    pred = model.predict(inp, batched=True)
    assert len(pred) == 2
    prob = model.predict_proba(inp, batched=True)
    assert prob.shape == (2, 2)
    pred = model.predict(inp[0], batched=False)
    assert isinstance(pred, int)
    prob = model.predict_proba(inp[0], batched=False)
    assert prob.shape == (2,)
    pred = model.pipeline("ew. getting ready for work")
    assert isinstance(pred, int) and pred in [0, 1]
