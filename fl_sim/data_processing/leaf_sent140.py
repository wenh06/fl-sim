from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import torch
import torch.utils.data as torchdata

from ..utils.const import CACHED_DATA_DIR
from ..models import nn as mnn
from ..models.utils import top_n_accuracy  # noqa: F401
from .fed_dataset import FedNLPDataset  # noqa: F401
from ._register import register_fed_dataset  # noqa: F401


__all__ = [
    "LeafSent140",
]


LEAF_SENT140_DATA_DIR = CACHED_DATA_DIR / "leaf_sent140"
LEAF_SENT140_DATA_DIR.mkdir(parents=True, exist_ok=True)


# @register_fed_dataset()
class LeafSent140(FedNLPDataset):
    """Federeated Sentiment140 dataset from Leaf.

    Sentiment140 dataset [1]_ is built from the tweets of Twitter
    and is used to perform sentiment analysis tasks. The Leaf library [2]_
    further processed the data.

    Parameters
    ----------
    datadir : Union[pathlib.Path, str], optional
        Directory to store data.
        If ``None``, use default directory.
    seed : int, default 0
        Random seed for data partitioning.
    **extra_config : dict, optional
        Extra configurations.

    References
    ----------
    .. [1] http://help.sentiment140.com
    .. [2] https://github.com/TalwalkarLab/leaf/tree/master/data/sent140

    """

    __name__ = "LeafSent140"

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        """Preload the dataset.

        Parameters
        ----------
        datadir : Union[pathlib.Path, str], optional
            Directory to store data.
            If ``None``, use default directory.

        Returns
        -------
        None

        """
        self.criterion = torch.nn.CrossEntropyLoss()
        raise NotImplementedError

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        """Get local dataloader at client `client_idx` or get the global dataloader.

        Parameters
        ----------
        train_bs : int, optional
            Batch size for training dataloader.
            If ``None``, use default batch size.
        test_bs : int, optional
            Batch size for testing dataloader.
            If ``None``, use default batch size.
        client_idx : int, optional
            Index of the client to get dataloader.
            If ``None``, get the dataloader containing all data.
            Usually used for centralized training.

        Returns
        -------
        train_dl : :class:`torch.utils.data.DataLoader`
            Training dataloader.
        test_dl : :class:`torch.utils.data.DataLoader`
            Testing dataloader.

        """
        raise NotImplementedError

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        """Evaluation using predictions and ground truth.

        Parameters
        ----------
        probs : torch.Tensor
            Predicted probabilities.
        truths : torch.Tensor
            Ground truth labels.

        Returns
        -------
        Dict[str, float]
            Evaluation results.

        """
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        """URL for downloading the dataset."""
        return "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """A set of candidate models."""
        return {
            "rnn": mnn.RNN_Sent140(),
        }

    @property
    def doi(self) -> List[str]:
        """DOI(s) related to the dataset."""
        return [
            "10.48550/ARXIV.1812.01097",  # LEAF
        ]
