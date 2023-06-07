"""
The dataset Sent140 used in the FedProx paper

References
----------
1. https://github.com/litian96/FedProx/tree/master/data/sent140
"""

import json
import re
import warnings
from pathlib import Path
from string import punctuation
from typing import Optional, Union, List, Tuple, Dict

import torch
import torch.utils.data as torchdata

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None
    warnings.warn(
        "Text preprocessing is better with Beautiful Soup 4. "
        "One can install it by `pip install beautifulsoup4`.",
        RuntimeWarning,
    )

from ..utils.const import CACHED_DATA_DIR
from ..models import nn as mnn
from ..models.word_embeddings import GloveEmbedding
from ..models.utils import top_n_accuracy
from .fed_dataset import FedNLPDataset


__all__ = [
    "FedProxSent140",
]


FEDPROX_SENT140_DATA_DIR = CACHED_DATA_DIR / "fedprox_sent140"
FEDPROX_SENT140_DATA_DIR.mkdir(parents=True, exist_ok=True)


class FedProxSent140(FedNLPDataset):
    """Federated Sentiment140 dataset used in FedProx paper.

    Sentiment140 dataset (ref. [1]_) is built from the tweets
    with positive and negative sentiment. FedProx (ref. [2]_)
    preprocessed the data and saved the data into json files.

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
    .. [2] https://github.com/litian96/FedProx/tree/master/data/sent140

    """

    __name__ = "FedProxSent140"

    def _preload(
        self,
        datadir: Optional[Union[str, Path]] = None,
        embedding_name: str = "glove.6B.50d",
    ) -> None:
        self.datadir = Path(datadir or FEDPROX_SENT140_DATA_DIR).expanduser().resolve()
        self.download_if_needed()

        self.embedding_name = embedding_name

        self.DEFAULT_BATCH_SIZE = 4
        self.DEFAULT_TRAIN_FILE = list((self.datadir / "train").glob("*.json"))[0]
        self.DEFAULT_TEST_FILE = list((self.datadir / "test").glob("*.json"))[0]

        self._EXAMPLE = "user_data"
        self._TEXT = "x"
        self._LABEL = "y"

        self._class_map = {
            "0": 0,
            "4": 1,
        }

        # data are dict with keys "users", "num_samples", "user_data"
        self._train_data_dict = json.loads(self.DEFAULT_TRAIN_FILE.read_text())
        self._test_data_dict = json.loads(self.DEFAULT_TEST_FILE.read_text())

        self.DEFAULT_TRAIN_CLIENTS_NUM = len(self._train_data_dict["users"])
        self.DEFAULT_TEST_CLIENTS_NUM = len(self._test_data_dict["users"])

        # find the raw max length of the text,
        # and remove URLs and leading "@user" in the text,
        # then remove leading and trailing punctuation
        raw_max_len = 0
        for user in self._train_data_dict["users"]:
            for idx in range(
                len(self._train_data_dict[self._EXAMPLE][user][self._TEXT])
            ):
                text = self._train_data_dict[self._EXAMPLE][user][self._TEXT][idx][
                    4
                ].strip()
                text = self._preprocess_text(text)
                self._train_data_dict[self._EXAMPLE][user][self._TEXT][idx][4] = text
                raw_max_len = max(raw_max_len, len(text.split()))
        for user in self._test_data_dict["users"]:
            for idx in range(len(self._test_data_dict[self._EXAMPLE][user])):
                text = self._test_data_dict[self._EXAMPLE][user][self._TEXT][idx][
                    4
                ].strip()
                text = self._preprocess_text(text)
                self._test_data_dict[self._EXAMPLE][user][self._TEXT][idx][4] = text
                raw_max_len = max(raw_max_len, len(text.split()))
        # print(f"raw_max_len: {raw_max_len}")

        self.max_len = min(raw_max_len * 2, 50)  # 25 in original repo

        try:
            self.tokenizer = GloveEmbedding.get_tokenizer(
                self.embedding_name, max_length=self.max_len
            )
        except FileNotFoundError:
            self.tokenizer = GloveEmbedding(self.embedding_name)._get_tokenizer(
                max_length=self.max_len
            )

        self.criterion = torch.nn.CrossEntropyLoss()

        self._client_ids_train = self._train_data_dict["users"]
        self._client_ids_test = self._test_data_dict["users"]

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self._client_ids_train
            test_ids = self._client_ids_test
        else:
            # get ids of single client
            train_ids = [self._client_ids_train[client_idx]]
            test_ids = [self._client_ids_test[client_idx]]

        train_x, train_y, test_x, test_y = [], [], [], []
        for user in train_ids:
            train_x.extend(
                [
                    item[4]
                    for item in self._train_data_dict[self._EXAMPLE][user][self._TEXT]
                ]
            )
            train_y.extend(self._train_data_dict[self._EXAMPLE][user][self._LABEL])
        for user in test_ids:
            test_x.extend(
                [
                    item[4]
                    for item in self._test_data_dict[self._EXAMPLE][user][self._TEXT]
                ]
            )
            test_y.extend(self._test_data_dict[self._EXAMPLE][user][self._LABEL])

        # tokenize to tensor
        train_x = self.tokenizer(train_x, return_tensors="pt")
        test_x = self.tokenizer(test_x, return_tensors="pt")
        train_y = torch.tensor([self._class_map[lb] for lb in train_y])
        test_y = torch.tensor([self._class_map[lb] for lb in test_y])

        train_ds = torchdata.TensorDataset(train_x, train_y)
        test_ds = torchdata.TensorDataset(test_x, test_y)

        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )
        test_dl = torchdata.DataLoader(
            dataset=test_ds,
            batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        return train_dl, test_dl

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        remove URLs and leading "@user" in the text,
        then remove leading and trailing punctuation
        """
        if BeautifulSoup is not None:
            text = BeautifulSoup(text, "html.parser").get_text()
        # remove the URLs
        text = re.sub(r"(^|\s*)https?://\S+(\s+|$)", "", text.strip())
        # remove the leading "@user"
        text = re.sub("^@\\S+\\s", "", text.strip())
        # remove leading or trailing punctuations
        text = re.sub("^[" + punctuation + "\\s]+", "", text.strip())
        text = re.sub("[\\s" + punctuation + "]+$", "", text.strip())
        return text.strip()

    def get_word_dict(self) -> Dict[str, int]:
        return self.tokenizer._tokenizer.get_vocab()

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        # https://drive.google.com/file/d/1pgHf4DUZkGI6q-NLjBzMawX5yn4Y40k0/view?usp=sharing
        return "https://www.dropbox.com/s/jbmubdtehkwade1/fedprox-sent140.zip?dl=1"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "rnn": mnn.RNN_Sent140(embedding=self.embedding_name),
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.1812.06127"]

    def view_sample(self, client_idx: int, sample_idx: int) -> None:
        if client_idx >= len(self._client_ids_train):
            raise ValueError(
                f"client_idx should be less than {len(self._client_ids_train)}"
            )
        user = self._client_ids_train[client_idx]
        raw_texts = (
            self._train_data_dict[self._EXAMPLE][user][self._TEXT]
            + self._test_data_dict[self._EXAMPLE][user][self._TEXT]
        )
        if sample_idx >= len(raw_texts):
            raise ValueError(f"sample_idx should be less than {len(raw_texts)}")
        text = raw_texts[sample_idx][4]
        label = (
            self._train_data_dict[self._EXAMPLE][user][self._LABEL]
            + self._test_data_dict[self._EXAMPLE][user][self._LABEL]
        )
        label = label[sample_idx]

        new_line = "\n" + "-" * 50 + "\n"
        print(f"Text:{new_line}{text}{new_line}")
        print(f"Label:{new_line}{label}")
