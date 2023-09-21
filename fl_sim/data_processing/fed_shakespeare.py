from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch  # noqa: F401
import torch.utils.data as torchdata

from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from ..utils.const import CACHED_DATA_DIR
from ._register import register_fed_dataset
from .fed_dataset import FedNLPDataset

__all__ = [
    "FedShakespeare",
]


FED_SHAKESPEARE_DATA_DIR = CACHED_DATA_DIR / "fed_shakespeare"
FED_SHAKESPEARE_DATA_DIR.mkdir(parents=True, exist_ok=True)


@register_fed_dataset()
class FedShakespeare(FedNLPDataset):
    """Federated Shakespeare dataset.

    Shakespeare dataset is built from the collective works of William Shakespeare.
    This dataset is used to perform tasks of next character prediction.
    FedML [1]_ loaded data from TensorFlow Federated (TFF) shakespeare load_data API [2]_
    and saved the unzipped data into hdf5 files.

    Data partition is the same as TFF, with the following statistics.

    +-------------+---------------+----------------+--------------+---------------+
    | DATASET     | TRAIN CLIENTS | TRAIN EXAMPLES | TEST CLIENTS | TEST EXAMPLES |
    +=============+===============+================+==============+===============+
    | SHAKESPEARE | 715           | 16,068         | 715          | 2356          |
    +-------------+---------------+----------------+--------------+---------------+

    Each client corresponds to a speaking role with at least two lines.

    Parameters
    ----------
    datadir : Union[str, pathlib.Path], optional
        The directory to store the dataset.
        If ``None``, use default directory.
    seed : int, default 0
        The random seed.
    **extra_config : dict, optional
        Extra configurations.

    References
    ----------
    .. [1] https://github.com/FedML-AI/FedML/tree/master/python/fedml/data/fed_shakespeare
    .. [2] https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data

    """

    SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
    # Vocabulary re-used from the Federated Learning for Text Generation tutorial.
    # https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
    CHAR_VOCAB = list("dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#'/37;?bfjnrvzBFJNRVZ\"&*.26:\naeimquyAEIMQUY]!%)-159\r")
    _pad = "<pad>"
    _bos = "<bos>"
    _eos = "<eos>"
    _oov = "<oov>"
    _words = [_pad] + CHAR_VOCAB + [_bos] + [_eos]
    word_dict = OrderedDict({w: i for i, w in enumerate(_words)})

    __name__ = "FedShakespeare"

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
        self.datadir = Path(datadir or FED_SHAKESPEARE_DATA_DIR).expanduser().resolve()

        self.DEFAULT_TRAIN_CLIENTS_NUM = 715
        self.DEFAULT_TEST_CLIENTS_NUM = 715
        self.DEFAULT_BATCH_SIZE = 4
        self.DEFAULT_TRAIN_FILE = "shakespeare_train.h5"
        self.DEFAULT_TEST_FILE = "shakespeare_test.h5"

        # group name defined by tff in h5 file
        self._EXAMPLE = "examples"
        self._SNIPPETS = "snippets"

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.download_if_needed()

        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        with h5py.File(str(train_file_path), "r") as train_h5, h5py.File(str(test_file_path), "r") as test_h5:
            self._client_ids_train = list(train_h5[self._EXAMPLE].keys())
            self._client_ids_test = list(test_h5[self._EXAMPLE].keys())

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
        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")
        train_ds = []
        test_ds = []

        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self._client_ids_train
            test_ids = self._client_ids_test
        else:
            # get ids of single client
            train_ids = [self._client_ids_train[client_idx]]
            test_ids = [self._client_ids_test[client_idx]]

        for client_id in train_ids:
            raw_train = train_h5[self._EXAMPLE][client_id][self._SNIPPETS][()]
            raw_train = [x.decode("utf8") for x in raw_train]
            train_ds.extend(self.preprocess(raw_train))
        for client_id in test_ids:
            raw_test = test_h5[self._EXAMPLE][client_id][self._SNIPPETS][()]
            raw_test = [x.decode("utf8") for x in raw_test]
            test_ds.extend(self.preprocess(raw_test))

        # split data
        train_x, train_y = FedShakespeare._split_target(train_ds)
        test_x, test_y = FedShakespeare._split_target(test_ds)
        train_ds = torchdata.TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
        test_ds = torchdata.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
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

        train_h5.close()
        test_h5.close()
        return train_dl, test_dl

    @staticmethod
    def _split_target(sequence_batch: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Split a N + 1 sequence into shifted-by-1 sequences for input and output."""
        sequence_batch = np.asarray(sequence_batch)
        input_text = sequence_batch[..., :-1]
        target_text = sequence_batch[..., 1:]
        return (input_text, target_text)

    def preprocess(self, sentences: Sequence[str], max_seq_len: Optional[int] = None) -> List[List[int]]:
        """Preprocess a list of sentences.

        Parameters
        ----------
        sentences : Sequence[str]
            List of sentences to be preprocessed.
        max_seq_len : int, optional
            Maximum sequence length.
            If ``None``, use default sequence length.

        Returns
        -------
        List[List[int]]
            List of tokenized sentences.

        """
        sequences = []
        if max_seq_len is None:
            max_seq_len = self.SEQUENCE_LENGTH

        def to_ids(sentence: str, num_oov_buckets: int = 1) -> Tuple[List[int]]:
            """Map list of sentence to list of ``[idx..]``
            and pad to ``max_seq_len + 1``.

            Parameters
            ----------
            sentence : str
                Sentence to be converted.
            num_oov_buckets : int, default 1
                The number of out of vocabulary buckets.

            Returns
            -------
            Tuple[List[int]]
                List of tokenized sentence.

            """
            tokens = [self.char_to_id(c) for c in sentence]
            tokens = [self.char_to_id(self._bos)] + tokens + [self.char_to_id(self._eos)]
            if len(tokens) % (max_seq_len + 1) != 0:
                pad_length = (-len(tokens)) % (max_seq_len + 1)
                tokens += list(repeat(self.char_to_id(self._pad), pad_length))
            return (tokens[i : i + max_seq_len + 1] for i in range(0, len(tokens), max_seq_len + 1))

        for sen in sentences:
            sequences.extend(to_ids(sen))
        return sequences

    def id_to_word(self, idx: int) -> str:
        """Convert an integer index to a character."""
        return self.words[idx]

    def char_to_id(self, char: str) -> int:
        """Convert a character to an integer index."""
        return self.word_dict.get(char, len(self.word_dict))

    @property
    def words(self) -> List[str]:
        """Get the word list."""
        return self._words

    def get_word_dict(self) -> Dict[str, int]:
        """Get the word dictionary."""
        return self.word_dict

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
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        """URL for downloading the dataset."""
        return "https://fedml.s3-us-west-1.amazonaws.com/shakespeare.tar.bz2"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """A set of candidate models."""
        return {
            "rnn": mnn.RNN_OriginalFedAvg(),
        }

    @property
    def doi(self) -> List[str]:
        """DOI(s) related to the dataset."""
        return [
            "10.48550/ARXIV.1812.06127",  # FedProx
            "10.48550/ARXIV.2007.13518",  # FedML
        ]

    def view_sample(self, client_idx: int, sample_idx: Optional[int] = None) -> None:
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
            raise ValueError(f"client_idx must be less than {len(self._client_ids_train)}")
        client_id = self._client_ids_train[client_idx]  # also test ids

        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")

        raw_train = train_h5[self._EXAMPLE][client_id][self._SNIPPETS][()]
        raw_train = [x.decode("utf8") for x in raw_train]
        raw_test = test_h5[self._EXAMPLE][client_id][self._SNIPPETS][()]
        raw_test = [x.decode("utf8") for x in raw_test]

        snippets = raw_train + raw_test

        new_line = "\n" + "-" * 50 + "\n"

        if sample_idx is not None:
            assert sample_idx < len(snippets), "sample_idx out of range"

        print(f"Client ID (Title):{new_line}{client_id}{new_line}")

        if sample_idx is None:
            print(f"Snippets:{new_line}{new_line.join([repr(x) for x in snippets])}")
        else:
            print(f"Snippet {sample_idx}:{new_line}{repr(snippets[sample_idx])}")

        train_h5.close()
        test_h5.close()
