"""
"""

from typing import Optional, List, Tuple, Dict

import numpy as np
import torch  # noqa: F401
import torch.utils.data as data

from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedDataset
from ._generate_synthetic import generate_synthetic


__all__ = [
    "FedSynthetic",
]


class FedSynthetic(FedDataset):

    __name__ = "FedSynthetic"

    def __init__(
        self,
        alpha: float,
        beta: float,
        iid: bool,
        num_clients: int,
        num_classes: int = 10,
        dimension: int = 60,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.iid = iid
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.dimension = dimension

        self._preload()

    def _preload(self, seed: int = 0) -> None:
        self.criterion = torch.nn.CrossEntropyLoss()
        self._data = generate_synthetic(
            alpha=self.alpha,
            beta=self.beta,
            iid=self.iid,
            num_clients=self.num_clients,
            num_classes=self.num_classes,
            dimension=self.dimension,
            seed=seed,
        )
        self.DEFAULT_BATCH_SIZE = 8
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
        """get local dataloader at client `client_idx` or get the global dataloader"""
        assert client_idx is None or 0 <= client_idx < self.num_clients
        if client_idx is None:
            train_X = np.concatenate([d["train_X"] for d in self._data], axis=0)
            train_y = np.concatenate([d["train_y"] for d in self._data], axis=0)
            test_X = np.concatenate([d["test_X"] for d in self._data], axis=0)
            test_y = np.concatenate([d["test_y"] for d in self._data], axis=0)
        else:
            train_X = self._data[client_idx]["train_X"]
            train_y = self._data[client_idx]["train_y"]
            test_X = self._data[client_idx]["test_X"]
            test_y = self._data[client_idx]["test_y"]

        train_bs = train_bs or self.DEFAULT_BATCH_SIZE
        if train_bs == -1:
            train_bs = len(train_X)
        train_dl = data.DataLoader(
            dataset=data.TensorDataset(
                torch.from_numpy(train_X), torch.from_numpy(train_y)
            ),
            batch_size=train_bs,
            shuffle=True,
            drop_last=False,
        )
        test_bs = test_bs or self.DEFAULT_BATCH_SIZE
        if test_bs == -1:
            test_bs = len(test_X)
        test_dl = data.DataLoader(
            dataset=data.TensorDataset(
                torch.from_numpy(test_X), torch.from_numpy(test_y)
            ),
            batch_size=test_bs,
            shuffle=True,
            drop_last=False,
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

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "alpha",
            "beta",
            "iid",
            "num_clients",
            "num_classes",
            "dimension",
        ]

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        return ""

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "mlp_d1": mnn.MLP(self.dimension, self.num_classes, ndim=0),
            "mlp_d2": mnn.MLP(
                self.dimension,
                self.num_classes,
                [
                    2 * self.dimension,
                ],
                ndim=0,
            ),
            "mlp_d3": mnn.MLP(
                self.dimension,
                self.num_classes,
                [
                    int(1.5 * self.dimension),
                    2 * self.dimension,
                ],
                ndim=0,
            ),
            "mlp_d4": mnn.MLP(
                self.dimension,
                self.num_classes,
                [
                    int(1.5 * self.dimension),
                    2 * self.dimension,
                    int(1.5 * self.dimension),
                ],
                ndim=0,
            ),
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.1812.06127"]
