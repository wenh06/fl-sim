"""
"""

from typing import Union, Optional, Dict, List

import einops
import numpy as np
import torch
from torch import Tensor


__all__ = [
    "CLFMixin",
    "REGMixin",
    "DiffMixin",
    "top_n_accuracy",
]


class CLFMixin(object):
    """Mixin class for classifiers."""

    __name__ = "CLFMixin"

    def predict_proba(
        self,
        input: Union[Tensor, np.ndarray],
        multi_label: bool = False,
        batched: bool = False,
    ) -> np.ndarray:
        """Predict probabilities for each class.

        Parameters
        ----------
        input : torch.Tensor or numpy.ndarray
            The input data.
        multi_label : bool, default False
            Whether the model is a multi-label classifier.
        batched : bool, default False
            Whether the input is batched.

        Returns
        -------
        proba : numpy.ndarray
            The predicted probabilities.

        """
        self.eval()
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).to(self.device)
        if not batched:
            input = input.unsqueeze(0)
        output = self.forward(input)
        if multi_label:
            proba = torch.sigmoid(output).cpu().detach().numpy()
        proba = torch.softmax(output, dim=-1).cpu().detach().numpy()
        if not batched:
            proba = proba.squeeze(0)
        return proba

    def predict(
        self,
        input: Union[Tensor, np.ndarray],
        thr: Optional[float] = None,
        class_map: Optional[Dict[int, str]] = None,
        batched: bool = False,
    ) -> list:
        """Predict the class labels.

        Parameters
        ----------
        input : torch.Tensor or numpy.ndarray
            The input data.
        thr : float, optional
            The threshold for multi-label classification.
            None for single-label classification.
        class_map : dict, optional
            The mapping from class index to class name.
        batched : bool, default False
            Whether the input is batched.

        Returns
        -------
        labels : list
            The predicted class labels.

        """
        proba = self.predict_proba(input, multi_label=thr is not None, batched=batched)
        if thr is None:
            output = proba.argmax(axis=-1).tolist()
            if class_map is not None:
                if batched:
                    output = [class_map[i] for i in output]
                else:
                    output = class_map[output]
            return output
        if batched:
            output = [[] for _ in range(input.shape[0])]
        else:
            output = [[]]
        if not batched:
            proba = proba[np.newaxis, ...]
        indices = np.where(proba > thr)
        if len(indices) > 2:
            raise ValueError(
                "multi-label classification is not supported for output of 3 dimensions or more"
            )
        for i, j in zip(*indices):
            output[i].append(j)
        for idx in range(len(output)):
            if len(output[idx]) == 0:
                output[idx] = [proba[idx].argmax()]
        if class_map is not None:
            output = [[class_map[i] for i in item] for item in output]
        if not batched:
            output = output[0]
        return output


class REGMixin(object):
    """Mixin for regressors."""

    __name__ = "REGMixin"

    def predict(self, input: Tensor) -> np.ndarray:
        """Predict the regression target.

        Parameters
        ----------
        input : torch.Tensor
            The input data.

        Returns
        -------
        output : numpy.ndarray
            The predicted regression target.

        """
        output = self.forward(input)
        return output.cpu().detach().numpy()


class DiffMixin(object):
    """Mixin for differences of two models."""

    def diff(
        self, other: object, norm: Optional[Union[str, int, float]] = None
    ) -> Union[float, List[Tensor]]:
        """Compute the difference between two models.

        Parameters
        ----------
        other : object
            Another model, which has the same structure as this one.
        norm : str or int or float, optional
            The norm to compute the difference.
            None for the raw difference.
            refer to :func:`torch.norm` for more details.

        Returns
        -------
        diff : float or list of torch.Tensor
            The difference.

        """
        assert isinstance(
            other, type(self)
        ), "the two models should have the same structure"
        if norm is not None:
            if norm == "inf":
                norm = float("inf")
            elif norm == "-inf":
                norm = -float("inf")
            assert isinstance(norm, (int, float)) or norm in ["nuc", "fro"], (
                "norm should be an int or float or one of "
                "'nuc' (nuclear norm) or 'fro' (Frobenius norm)"
            )
        try:
            if norm is not None:
                diff = [
                    torch.norm(p1 - p2, p=norm).item()
                    for p1, p2 in zip(self.parameters(), other.parameters())
                ]
            else:
                diff = [
                    p1.data - p2.data
                    for p1, p2 in zip(self.parameters(), other.parameters())
                ]
        except RuntimeError as e:
            if norm == "nuc" and "Expected a tensor with 2 dimensions" in str(e):
                raise ValueError(
                    "nuclear norm is not supported for the current model structure"
                ) from e
            elif "must match the size" in str(e):
                raise ValueError("the two models should have the same structure") from e
            else:
                raise e
        if norm in ["nuc", "fro"]:
            diff = np.sqrt(np.sum([d**2 for d in diff]))
        elif norm == float("inf"):
            diff = np.max([d for d in diff])
        elif norm == -float("inf"):
            diff = np.min([d for d in diff])
        elif isinstance(norm, (int, float)):
            # L_p norm for p finite
            diff = np.sum([d**norm for d in diff]) ** (1 / norm)
        return diff


def top_n_accuracy(preds: Tensor, labels: Tensor, n: int = 1) -> float:
    """Top-n accuracy.

    Parameters
    ----------
    preds : torch.Tensor
        Shape ``(batch_size, n_classes)`` or ``(batch_size, n_classes, d_1, ..., d_n)``.
    labels : torch.Tensor
        Shape ``(batch_size,)`` or ``(batch_size, d_1, ..., d_n)``.

    Returns
    -------
    float
        The top-n accuracy.

    """
    assert preds.shape[0] == labels.shape[0]
    batch_size, n_classes, *extra_dims = preds.shape
    _, indices = torch.topk(
        preds, n, dim=1
    )  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
    pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
    pattern = f"batch_size {pattern} -> batch_size n {pattern}"
    correct = torch.sum(indices == einops.repeat(labels, pattern, n=n))
    acc = correct.item() / preds.shape[0]
    for d in extra_dims:
        acc = acc / d
    return acc
