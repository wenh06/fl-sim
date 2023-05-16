"""
dataset readers, for datasets from LIBSVM library
"""

import posixpath
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import torch
import torch.utils.data as data
from deprecate_kwargs import deprecate_kwargs

from ..utils.misc import CACHED_DATA_DIR
from ..utils._download_data import http_get
from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedDataset


__all__ = [
    "FedLibSVMDataset",
    "libsvmread",
]


_libsvm_domain = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"

# NOT fully listed
_libsvm_datasets = {
    f"a{i}a": [f"binary/a{i}a", f"binary/a{i}a.t"] for i in range(1, 10)
}
_libsvm_datasets.update(
    {f"w{i}a": [f"binary/w{i}a", f"binary/w{i}a.t"] for i in range(1, 9)}
)
_libsvm_datasets = {
    k: [posixpath.join(_libsvm_domain, item) for item in v]
    for k, v in _libsvm_datasets.items()
}


class FedLibSVMDataset(FedDataset):

    __name__ = "FedLibSVMDataset"

    @deprecate_kwargs([["criterion_name", "criterion"]])
    def __init__(
        self,
        dataset_name: str,
        num_clients: int,
        iid: bool = True,
        criterion: str = "svm",
    ) -> None:
        self.dataset_name = dataset_name
        assert self.dataset_name in _libsvm_datasets, (
            f"dataset {self.dataset_name} not supported, "
            f"supported datasets: {list(_libsvm_datasets.keys())}"
        )
        self.num_clients = num_clients
        self.iid = iid
        if not self.iid:
            # ``non_iid_partition_with_dirichlet_distribution`` is too slow
            raise NotImplementedError("non-iid not implemented yet")
        self.criterion_name = criterion.lower()
        self.datadir = CACHED_DATA_DIR / "libsvm_datasets" / dataset_name

        self.criterion = None
        self._data = None
        self.__num_features, self.__num_classes = None, None
        self._preload()

    def _preload(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.datadir.mkdir(parents=True, exist_ok=True)
        self.download_if_needed()
        self.criterion = self.criteria_mapping[self.criterion_name]
        train_X, train_y = libsvmread(self.datadir / self.dataset_name, toarray=True)
        shuffled_indices = np.arange(len(train_y))
        rng.shuffle(shuffled_indices)
        train_X = train_X[shuffled_indices]
        train_y = train_y[shuffled_indices]
        test_X, test_y = libsvmread(
            self.datadir / f"{self.dataset_name}.t", toarray=True
        )
        shuffled_indices = np.arange(len(test_y))
        rng.shuffle(shuffled_indices)
        test_X = test_X[shuffled_indices]
        test_y = test_y[shuffled_indices]

        self.__num_features = train_X.shape[1]
        self.__num_classes = len(np.unique(train_y))

        # do partition
        min_gap = int(
            np.ceil(self.num_clients * test_X.shape[0] / train_X.shape[0]) + 1
        )
        train_split_indices = np.sort(
            rng.choice(
                train_X.shape[0] - self.num_clients * min_gap,
                self.num_clients,
                replace=False,
            )
        ) + min_gap * np.arange(self.num_clients)
        train_split_indices = np.append(train_split_indices, train_X.shape[0])
        test_split_indices = (
            train_split_indices / train_X.shape[0] * test_X.shape[0]
        ).astype(int)
        test_split_indices[-1] = test_X.shape[0]

        self._data = [
            {
                "train_X": train_X[train_split_indices[i] : train_split_indices[i + 1]],
                "train_y": train_y[train_split_indices[i] : train_split_indices[i + 1]],
                "test_X": test_X[test_split_indices[i] : test_split_indices[i + 1]],
                "test_y": test_y[test_split_indices[i] : test_split_indices[i + 1]],
            }
            for i in range(self.num_clients)
        ]

        self.DEFAULT_BATCH_SIZE = -1
        self.DEFAULT_TRAIN_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TEST_CLIENTS_NUM = self.num_clients

    def reset_seed(self, seed: int) -> None:
        self._preload(seed)

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        assert client_idx is None or 0 <= client_idx < self.num_clients
        if client_idx is None:
            train_X = np.concatenate(
                [self._data[i]["train_X"] for i in range(self.num_clients)]
            )
            train_y = np.concatenate(
                [self._data[i]["train_y"] for i in range(self.num_clients)]
            )
            test_X = np.concatenate(
                [self._data[i]["test_X"] for i in range(self.num_clients)]
            )
            test_y = np.concatenate(
                [self._data[i]["test_y"] for i in range(self.num_clients)]
            )
        else:
            train_X = self._data[client_idx]["train_X"]
            train_y = self._data[client_idx]["train_y"]
            test_X = self._data[client_idx]["test_X"]
            test_y = self._data[client_idx]["test_y"]

        train_bs = train_bs or self.DEFAULT_BATCH_SIZE
        if train_bs == -1:
            train_bs = len(train_X)
        train_dl = data.DataLoader(
            data.TensorDataset(
                torch.from_numpy(train_X).float(),
                torch.from_numpy(train_y).long(),
            ),
            batch_size=train_bs,
            shuffle=True,
        )
        test_bs = test_bs or self.DEFAULT_BATCH_SIZE
        if test_bs == -1:
            test_bs = len(test_X)
        test_dl = data.DataLoader(
            data.TensorDataset(
                torch.from_numpy(test_X).float(),
                torch.from_numpy(test_y).long(),
            ),
            batch_size=test_bs,
            shuffle=False,
        )
        return train_dl, test_dl

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
            self.num_clients,
            train_data_num,
            train_data_global,
            test_data_global,
            local_data_num,
            train_data_local,
            test_data_local,
            self.num_classes,
        )
        return retval

    def load_partition_data(self, batch_size: Optional[int] = None) -> tuple:
        """partition data into all local clients"""
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(self.num_clients):
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
            self.num_clients,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.num_classes,
        )

        return retval

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        raise posixpath.dirname(_libsvm_datasets[self.dataset_name][0]) + ".html"

    def download_if_needed(self) -> None:
        for url in _libsvm_datasets[self.dataset_name]:
            if (self.datadir / posixpath.basename(url)).exists():
                continue
            http_get(url, self.datadir, extract=False)

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "svm": mnn.SVC(self.num_features, self.num_classes),
            "lr": mnn.LogisticRegression(self.num_features, self.num_classes),
        }

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(_libsvm_datasets.keys())

    @classmethod
    def list_all_libsvm_datasets(cls) -> pd.DataFrame:
        return pd.read_html(_libsvm_domain)[0]

    @property
    def criteria_mapping(self) -> Dict[str, torch.nn.Module]:
        return {
            "svm": torch.nn.MultiMarginLoss(),
            "svr": torch.nn.MSELoss(),
            "lr": torch.nn.CrossEntropyLoss(),
            "logistic_regression": torch.nn.CrossEntropyLoss(),
        }

    @property
    def num_features(self) -> int:
        return self.__num_features

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    def extra_repr_keys(self) -> List[str]:
        return ["dataset_name", "iid", "num_clients"]

    @property
    def doi(self) -> List[str]:
        return ["10.1145/1961189.1961199"]


def libsvmread(
    fp: Union[str, Path], multilabel: bool = False, toarray: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = load_svmlight_file(
        str(fp), multilabel=multilabel, dtype=np.float32
    )
    if toarray:
        features = features.toarray()
    return features, labels
