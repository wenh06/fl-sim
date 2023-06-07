"""
The dataset Sent140 used in the FedProx paper

References
----------
1. https://github.com/litian96/FedProx/tree/master/data/sent140
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import torch
import torch.utils.data as torchdata

from ..utils.const import CACHED_DATA_DIR
from ..models import nn as mnn
from ..models.utils import top_n_accuracy  # noqa: F401
from .fed_dataset import FedNLPDataset  # noqa: F401


__all__ = [
    "LeafSent140",
]


LEAF_SENT140_DATA_DIR = CACHED_DATA_DIR / "leaf_sent140"
LEAF_SENT140_DATA_DIR.mkdir(parents=True, exist_ok=True)


class LeafSent140(FedNLPDataset):
    """Federeated Sentiment140 dataset from Leaf.

    Sentiment140 dataset (ref. [1]_) is built from the tweets of Twitter
    and is used to perform sentiment analysis tasks. The Leaf library (ref. [2]_)
    further processed the data.

    Parameters
    ----------
    datadir : Union[Path, str], optional
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
        self.criterion = torch.nn.CrossEntropyLoss()
        raise NotImplementedError

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        raise NotImplementedError

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        return "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "rnn": mnn.RNN_Sent140(),
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.1812.01097"]
