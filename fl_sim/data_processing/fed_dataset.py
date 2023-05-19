"""
abstract base classes of federated dataset provided by FedML, and more
"""

import random
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from string import punctuation
from typing import Optional, Union, List, Tuple, Dict, Iterable, Sequence, Callable

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from datasets import (
    Dataset as HFD,
    NamedSplit as HFNS,
    load_dataset as HFD_load_dataset,
)
from bib_lookup import CitationMixin
from PIL import Image
from torch_ecg.utils import ReprMixin

from ..utils.const import CACHED_DATA_DIR
from ..utils._download_data import download_if_needed


__all__ = [
    "FedDataset",
    "FedVisionDataset",
    "FedNLPDataset",
    "NLPDataset",
]


class FedDataset(ReprMixin, CitationMixin, ABC):

    __name__ = "FedDataset"

    @abstractmethod
    def get_dataloader(
        self,
        train_bs: int,
        test_bs: int,
        client_idx: Optional[int] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError

    @abstractmethod
    def _preload(self, datadir: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_partition_data_distributed(
        self, process_id: int, batch_size: Optional[int] = None
    ) -> tuple:
        """get local dataloader at client `process_id` or get global dataloader"""
        raise NotImplementedError

    @abstractmethod
    def load_partition_data(self, batch_size: Optional[int] = None) -> tuple:
        """partition data into all local clients"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "datadir",
        ]

    @property
    @abstractmethod
    def url(self) -> str:
        raise NotImplementedError

    def download_if_needed(self) -> None:
        if self.url:
            if self.datadir is None:
                dst_dir = CACHED_DATA_DIR
            elif self.datadir.exists() and len(list(self.datadir.iterdir())) > 0:
                print("data dir exists, skip downloading")
                return
            else:
                # dst_dir = self.datadir.parent
                pass
            download_if_needed(self.url, self.datadir, extract=True)
            return
        print("No url for downloading data")

    @property
    @abstractmethod
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        raise NotImplementedError

    @property
    def data_parts(self) -> List[str]:
        return ["train", "val"]

    @property
    @abstractmethod
    def doi(self) -> Union[str, List[str]]:
        raise NotImplementedError


class FedVisionDataset(FedDataset, ABC):

    __name__ = "FedVisionDataset"

    def __init__(self, datadir: Optional[Union[Path, str]] = None) -> None:
        self.datadir = Path(datadir) if datadir is not None else None

        self.DEFAULT_TRAIN_CLIENTS_NUM = None
        self.DEFAULT_TEST_CLIENTS_NUM = None
        self.DEFAULT_BATCH_SIZE = None
        self.DEFAULT_TRAIN_FILE = None
        self.DEFAULT_TEST_FILE = None

        # group name defined by tff in h5 file
        self._EXAMPLE = "examples"
        self._IMGAE = "image"
        self._LABEL = "label"

        self._preload(datadir)

        assert all(
            [
                self.criterion is not None,
                self.datadir is not None,
                self.DEFAULT_TRAIN_CLIENTS_NUM is not None,
                self.DEFAULT_TEST_CLIENTS_NUM is not None,
                self.DEFAULT_BATCH_SIZE is not None,
                self.DEFAULT_TRAIN_FILE is not None,
                self.DEFAULT_TEST_FILE is not None,
            ]
        )

    @abstractmethod
    def get_dataloader(
        self,
        train_bs: int,
        test_bs: int,
        client_idx: Optional[int] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError

    @abstractmethod
    def _preload(self, datadir: Optional[str] = None) -> None:
        raise NotImplementedError

    def load_partition_data_distributed(
        self, process_id: int, batch_size: Optional[int] = None
    ) -> tuple:
        """get local dataloader at client `process_id` or get global dataloader"""
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.get_dataloader(
                _batch_size, _batch_size
            )
            train_data_num = len(train_data_global.dataset)
            test_data_num = len(test_data_global.dataset)
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_local, test_data_local = self.get_dataloader(
                _batch_size, _batch_size, process_id - 1
            )
            train_data_num = local_data_num = len(train_data_local.dataset)
            train_data_global = None
            test_data_global = None
        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num,
            train_data_global,
            test_data_global,
            local_data_num,
            train_data_local,
            test_data_local,
            self.n_class,
        )
        return retval

    def load_partition_data(self, batch_size: Optional[int] = None) -> tuple:
        """partition data into all local clients"""
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = self.get_dataloader(
                _batch_size, _batch_size, client_idx
            )
            local_data_num = len(train_data_local.dataset)
            data_local_num_dict[client_idx] = local_data_num
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        # global dataset
        train_data_global = data.DataLoader(
            data.ConcatDataset(
                list(dl.dataset for dl in list(train_data_local_dict.values()))
            ),
            batch_size=_batch_size,
            shuffle=True,
        )
        train_data_num = len(train_data_global.dataset)

        test_data_global = data.DataLoader(
            data.ConcatDataset(
                list(
                    dl.dataset
                    for dl in list(test_data_local_dict.values())
                    if dl is not None
                )
            ),
            batch_size=_batch_size,
            shuffle=True,
        )
        test_data_num = len(test_data_global.dataset)

        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.n_class,
        )

        return retval

    @property
    def n_class(self) -> int:
        return self._n_class

    @staticmethod
    def show_image(tensor: torch.Tensor) -> Image.Image:
        assert tensor.ndim == 3
        return transforms.ToPILImage()(tensor)

    @property
    @abstractmethod
    def label_map(self) -> dict:
        raise NotImplementedError

    def get_class(self, label: torch.Tensor) -> str:
        return self.label_map[label.item()]

    def get_classes(self, labels: torch.Tensor) -> List[str]:
        return [self.label_map[lb] for lb in labels.cpu().numpy()]


class FedNLPDataset(FedDataset, ABC):

    __name__ = "FedNLPDataset"

    def __init__(self, datadir: Optional[Union[str, Path]] = None) -> None:
        self.datadir = Path(datadir) if datadir is not None else None

        self.DEFAULT_TRAIN_CLIENTS_NUM = None
        self.DEFAULT_TEST_CLIENTS_NUM = None
        self.DEFAULT_BATCH_SIZE = None
        self.DEFAULT_TRAIN_FILE = None
        self.DEFAULT_TEST_FILE = None

        self._preload(datadir)

        assert all(
            [
                self.criterion is not None,
                self.datadir is not None,
                self.DEFAULT_TRAIN_CLIENTS_NUM is not None,
                self.DEFAULT_TEST_CLIENTS_NUM is not None,
                self.DEFAULT_BATCH_SIZE is not None,
                self.DEFAULT_TRAIN_FILE is not None,
                self.DEFAULT_TEST_FILE is not None,
            ]
        )

    @abstractmethod
    def get_dataloader(
        self,
        train_bs: int,
        test_bs: int,
        client_idx: Optional[int] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError

    def load_partition_data_distributed(
        self, process_id: int, batch_size: Optional[int] = None
    ) -> tuple:
        """get local dataloader at client `process_id` or get global dataloader"""
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.get_dataloader(
                batch_size, batch_size
            )
            train_data_num = len(train_data_global.dataset)
            test_data_num = len(test_data_global.dataset)
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_local, test_data_local = self.get_dataloader(
                batch_size, batch_size, process_id - 1
            )
            train_data_num = local_data_num = len(train_data_local.dataset)
            train_data_global = None
            test_data_global = None

        VOCAB_LEN = len(self.get_word_dict()) + 1

        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num,
            train_data_global,
            test_data_global,
            local_data_num,
            train_data_local,
            test_data_local,
            VOCAB_LEN,
        )

        return retval

    def load_partition_data(self, batch_size: Optional[int] = None) -> tuple:
        """partition data into all local clients"""
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE

        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = self.get_dataloader(
                batch_size, batch_size, client_idx
            )
            local_data_num = len(train_data_local.dataset)
            data_local_num_dict[client_idx] = local_data_num
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        # global dataset
        train_data_global = data.DataLoader(
            data.ConcatDataset(
                list(dl.dataset for dl in list(train_data_local_dict.values()))
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        train_data_num = len(train_data_global.dataset)

        test_data_global = data.DataLoader(
            data.ConcatDataset(
                list(
                    dl.dataset
                    for dl in list(test_data_local_dict.values())
                    if dl is not None
                )
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_data_num = len(test_data_global.dataset)

        VOCAB_LEN = len(self.get_word_dict()) + 1

        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            VOCAB_LEN,
        )

        return retval

    @abstractmethod
    def get_word_dict(self) -> Dict[str, int]:
        raise NotImplementedError


class NLPDataset(data.Dataset, ReprMixin):

    __name__ = "NLPDataset"

    def __init__(
        self,
        dataset: List[tuple],
        input_columns: List[str] = ["text"],
        label_map: Optional[Dict[int, int]] = None,
        label_names: Optional[List[str]] = None,
        output_scale_factor: Optional[float] = None,
        shuffle: bool = False,
        max_len: Optional[int] = 512,
    ) -> None:
        self._dataset = dataset
        self._name = None
        self.input_columns = input_columns
        self.label_map = label_map
        self.label_names = label_names
        if self.label_map and self.label_names:
            # If labels are remapped, the label names have to be remapped as well.
            self.label_names = [
                self.label_names[self.label_map[i]] for i in self.label_map
            ]
        self.shuffled = shuffle
        self.output_scale_factor = output_scale_factor

        if shuffle:
            random.shuffle(self._dataset)

        self.max_len = max_len

    def _format_as_dict(self, example: tuple) -> tuple:
        output = example[1]
        if self.label_map:
            output = self.label_map[output]
        if self.output_scale_factor:
            output = output / self.output_scale_factor

        if isinstance(example[0], str):
            if len(self.input_columns) != 1:
                raise ValueError(
                    "Mismatch between the number of columns in `input_columns` "
                    "and number of columns of actual input."
                )
            input_dict = OrderedDict(
                [(self.input_columns[0], self.clip_text(example[0]))]
            )
        else:
            if len(self.input_columns) != len(example[0]):
                raise ValueError(
                    "Mismatch between the number of columns in `input_columns` "
                    "and number of columns of actual input."
                )
            input_dict = OrderedDict(
                [
                    (c, self.clip_text(example[0][i]))
                    for i, c in enumerate(self.input_columns)
                ]
            )
        return input_dict, output

    def shuffle(self) -> None:
        random.shuffle(self._dataset)
        self.shuffled = True

    def filter_by_labels_(self, labels_to_keep: Iterable[int]) -> None:
        """
        Filter items by their labels for classification datasets. Performs
        in-place filtering.

        Parameters
        ----------
        labels_to_keep:
            Set, tuple, list, or iterable of integers representing labels.
        """
        if not isinstance(labels_to_keep, set):
            labels_to_keep = set(labels_to_keep)
        self._dataset = filter(lambda x: x[1] in labels_to_keep, self._dataset)

    def __getitem__(self, i: Union[slice, int]) -> Union[tuple, List[tuple]]:
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_as_dict(self._dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_as_dict(ex) for ex in self._dataset[i]]

    def __len__(self):
        """Returns the size of dataset."""
        return len(self._dataset)

    @staticmethod
    def from_huggingface_dataset(
        ds: Union[str, HFD], split: Optional[HFNS] = None, max_len: Optional[int] = 512
    ) -> "NLPDataset":
        if isinstance(ds, str):
            _ds = HFD_load_dataset(ds, split=split)
        else:
            _ds = ds
        if isinstance(_ds.column_names, dict):
            sets = list(_ds.column_names.keys())
            column_names = _ds.column_names[sets[0]]
        else:
            sets = []
            column_names = _ds.column_names
        input_columns, output_column = NLPDataset._split_dataset_columns(column_names)

        if sets:
            ret_ds = NLPDataset(
                [
                    (NLPDataset._gen_input(row, input_columns), row[output_column])
                    for s in sets
                    for row in _ds[s]
                ],
                input_columns=input_columns,
                max_len=max_len,
            )
        else:
            ret_ds = NLPDataset(
                [
                    (NLPDataset._gen_input(row, input_columns), row[output_column])
                    for row in _ds
                ],
                input_columns=input_columns,
                max_len=max_len,
            )
            ret_ds._name = _ds.info.builder_name
        return ret_ds

    def clip_text(self, text: str) -> str:
        if self.max_len is None:
            return text
        inds = [
            m.start()
            for m in re.finditer(f"[{punctuation}]", text)
            if m.start() < self.max_len
        ]
        if len(inds) == 0:
            return text[: self.max_len]
        return text[: inds[-1]]

    @property
    def dataset_name(self) -> str:
        return self._name

    def extra_repr_keys(self) -> List[str]:
        if self.dataset_name is not None:
            return ["dataset_name"]
        return super().extra_repr_keys()

    @staticmethod
    def _gen_input(row: dict, input_columns: Tuple[str]) -> Union[Tuple[str, ...], str]:
        if len(input_columns) == 1:
            return row[input_columns[0]]
        return tuple(row[c] for c in input_columns)

    @staticmethod
    def _split_dataset_columns(
        column_names: Sequence[str],
    ) -> Tuple[Tuple[str, ...], str]:
        """Common schemas for datasets found in huggingface datasets hub."""
        _column_names = set(column_names)
        if {"premise", "hypothesis", "label"} <= _column_names:
            input_columns = ("premise", "hypothesis")
            output_column = "label"
        elif {"question", "sentence", "label"} <= _column_names:
            input_columns = ("question", "sentence")
            output_column = "label"
        elif {"sentence1", "sentence2", "label"} <= _column_names:
            input_columns = ("sentence1", "sentence2")
            output_column = "label"
        elif {"question1", "question2", "label"} <= _column_names:
            input_columns = ("question1", "question2")
            output_column = "label"
        elif {"question", "sentence", "label"} <= _column_names:
            input_columns = ("question", "sentence")
            output_column = "label"
        elif {"text", "label"} <= _column_names:
            input_columns = ("text",)
            output_column = "label"
        elif {"sentence", "label"} <= _column_names:
            input_columns = ("sentence",)
            output_column = "label"
        elif {"document", "summary"} <= _column_names:
            input_columns = ("document",)
            output_column = "summary"
        elif {"content", "summary"} <= _column_names:
            input_columns = ("content",)
            output_column = "summary"
        elif {"label", "review"} <= _column_names:
            input_columns = ("review",)
            output_column = "label"
        else:
            raise ValueError(
                f"Unsupported dataset column_names {_column_names}. Try passing your own `dataset_columns` argument."
            )

        return input_columns, output_column

    def to_tensor_dataset(
        self,
        tokenizer: Callable[[Union[str, Sequence[str]]], torch.Tensor],
        labels_to_keep: Optional[Iterable[int]] = None,
    ) -> data.TensorDataset:
        """
        CAUTION: This method is not tested yet.
        """
        assert (
            self.label_map is not None
        ), "Label map must be set before converting to tensor dataset."
        if labels_to_keep is not None:
            self.filter_labels(labels_to_keep)
        X, y = {c: [] for c in self.input_columns}, []
        for ex in self:
            for c in self.input_columns:
                X[c].append(tokenizer(self.clip_text(ex[0][c])))
            y.append(ex[1])
        for c in self.input_columns:
            X[c] = tokenizer(X[c], return_tensors="pt")
        y = torch.tensor(y)
        return data.TensorDataset(*(X[c] for c in self.input_columns), y)
