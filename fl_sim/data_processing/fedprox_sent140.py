import json
import re
from pathlib import Path
from string import punctuation
from typing import Optional, Union, List, Tuple, Dict

import torch
import torch.utils.data as torchdata
from bs4 import BeautifulSoup

from ..utils.const import CACHED_DATA_DIR
from ..utils._download_data import url_is_reachable
from ..models import nn as mnn
from ..models.word_embeddings import GloveEmbedding
from ..models.utils import top_n_accuracy
from .fed_dataset import FedNLPDataset
from ._register import register_fed_dataset


__all__ = [
    "FedProxSent140",
]


FEDPROX_SENT140_DATA_DIR = CACHED_DATA_DIR / "fedprox_sent140"
FEDPROX_SENT140_DATA_DIR.mkdir(parents=True, exist_ok=True)


@register_fed_dataset()
class FedProxSent140(FedNLPDataset):
    """Federated Sentiment140 dataset used in FedProx paper.

    Sentiment140 dataset [1]_ is built from the tweets
    with positive and negative sentiment. FedProx [2]_
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
        embedding_name: str = "glove.6B.300d",
    ) -> None:
        """Preload the dataset.

        Parameters
        ----------
        datadir : Union[pathlib.Path, str], optional
            Directory to store data.
            If ``None``, use default directory.
        embedding_name : str, default "glove.6B.300d"
            Name of the word embedding to use.

        Returns
        -------
        None

        """
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

        # try:
        #     self.tokenizer = GloveEmbedding.get_tokenizer(
        #         self.embedding_name, max_length=self.max_len
        #     )
        # except FileNotFoundError:
        #     self.tokenizer = GloveEmbedding(self.embedding_name)._get_tokenizer(
        #         max_length=self.max_len
        #     )
        glove_embedding = GloveEmbedding(self.embedding_name)
        self.word_embeddings = glove_embedding.get_embedding_layer(freeze=True)
        self.tokenizer = glove_embedding.get_tokenizer(max_length=self.max_len)
        del glove_embedding

        self.criterion = torch.nn.CrossEntropyLoss()

        self._client_ids_train = self._train_data_dict["users"]
        self._client_ids_test = self._test_data_dict["users"]

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
        train_x = self.word_embeddings(self.tokenizer(train_x, return_tensors="pt"))
        test_x = self.word_embeddings(self.tokenizer(test_x, return_tensors="pt"))
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
        """Remove URLs and leading "@user" in the text,
        then remove leading and trailing punctuation.
        """
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
        # https://drive.google.com/file/d/1pgHf4DUZkGI6q-NLjBzMawX5yn4Y40k0/view?usp=sharing
        if url_is_reachable("https://www.dropbox.com"):
            return "https://www.dropbox.com/s/jbmubdtehkwade1/fedprox-sent140.zip?dl=1"
        else:
            return "https://deep-psp.tech/Data/FL/fedprox-sent140.zip"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """A set of candidate models."""
        return {
            "rnn": mnn.RNN_Sent140_LITE(embed_dim=self.word_embeddings.dim),
        }

    @property
    def doi(self) -> List[str]:
        """DOIs related to the dataset."""
        return [
            "10.48550/ARXIV.1812.01097",  # LEAF
            "10.48550/ARXIV.1812.06127",  # FedProx
        ]

    def view_sample(self, client_idx: int, sample_idx: int) -> None:
        """View a sample from the dataset.

        Parameters
        ----------
        client_idx : int
            Index of the client on which the sample is located.
        sample_idx : int
            Index of the sample in the client.

        Returns
        -------
        None

        """
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
