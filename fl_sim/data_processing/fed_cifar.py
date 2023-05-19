"""
federated Cifar10, Cifar100
"""

from pathlib import Path
from typing import Optional, Union, List, Callable, Tuple, Dict, Sequence

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from ..utils.const import CACHED_DATA_DIR, CIFAR100_FINE_LABEL_MAP, CIFAR10_LABEL_MAP
from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset


__all__ = [
    "FedCIFAR",
    "FedCIFAR100",
]


FED_CIFAR_DATA_DIRS = {
    n_class: (CACHED_DATA_DIR / f"fed_cifar{n_class}")
    for n_class in [
        10,
        100,
    ]
}
for n_class in [
    10,
    100,
]:
    FED_CIFAR_DATA_DIRS[n_class].mkdir(parents=True, exist_ok=True)


class FedCIFAR(FedVisionDataset):
    """
    most methods in this class are modified from FedML
    """

    __name__ = "FedCIFAR"

    def __init__(
        self, n_class: int = 100, datadir: Optional[Union[str, Path]] = None
    ) -> None:
        self._n_class = n_class
        assert self.n_class in [
            100,  # 10 not implemented
        ]
        super().__init__(
            datadir=Path(datadir or FED_CIFAR_DATA_DIRS[n_class]).expanduser().resolve()
        )

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        self.DEFAULT_TRAIN_CLIENTS_NUM = 500
        self.DEFAULT_TEST_CLIENTS_NUM = 100
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = f"fed_cifar{self.n_class}_train.h5"
        self.DEFAULT_TEST_FILE = f"fed_cifar{self.n_class}_test.h5"

        # group name defined by tff in h5 file
        self._EXAMPLE = "examples"
        self._IMGAE = "image"
        self._LABEL = "label"

        self.criterion = torch.nn.CrossEntropyLoss()

        self.download_if_needed()

        # client id list
        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        with h5py.File(str(train_file_path), "r") as train_h5, h5py.File(
            str(test_file_path), "r"
        ) as test_h5:
            self._client_ids_train = list(train_h5[self._EXAMPLE].keys())
            self._client_ids_test = list(test_h5[self._EXAMPLE].keys())

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """get local dataloader at client `client_idx` or get the global dataloader"""
        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")
        train_x, train_y, test_x, test_y = [], [], [], []

        # load data in numpy format from h5 file
        if client_idx is None:
            train_x = np.vstack(
                [
                    train_h5[self._EXAMPLE][client_id][self._IMGAE][()]
                    for client_id in self._client_ids_train
                ]
            )
            train_y = np.concatenate(
                [
                    train_h5[self._EXAMPLE][client_id][self._LABEL][()]
                    for client_id in self._client_ids_train
                ]
            )
            test_x = np.vstack(
                [
                    test_h5[self._EXAMPLE][client_id][self._IMGAE][()]
                    for client_id in self._client_ids_test
                ]
            )
            test_y = np.concatenate(
                [
                    test_h5[self._EXAMPLE][client_id][self._LABEL][()]
                    for client_id in self._client_ids_test
                ]
            )
        else:
            client_id_train = self._client_ids_train[client_idx]
            train_x = np.vstack(
                [train_h5[self._EXAMPLE][client_id_train][self._IMGAE][()]]
            )
            train_y = np.concatenate(
                [train_h5[self._EXAMPLE][client_id_train][self._LABEL][()]]
            )
            if client_idx <= len(self._client_ids_test) - 1:
                client_id_test = self._client_ids_test[client_idx]
                test_x = np.vstack(
                    [train_h5[self._EXAMPLE][client_id_test][self._IMGAE][()]]
                )
                test_y = np.concatenate(
                    [train_h5[self._EXAMPLE][client_id_test][self._LABEL][()]]
                )

        # preprocess
        transform = _data_transforms_fed_cifar(train=True)
        train_x = transform(
            torch.div(torch.from_numpy(train_x).permute(0, 3, 1, 2), 255.0)
        )
        train_y = torch.from_numpy(train_y)
        if len(test_x) != 0:
            transform = _data_transforms_fed_cifar(train=False)
            test_x = transform(
                torch.div(torch.from_numpy(test_x).permute(0, 3, 1, 2), 255.0)
            )
            test_y = torch.from_numpy(test_y)
            pass

        # generate dataloader
        train_ds = data.TensorDataset(train_x, train_y)
        train_dl = data.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        if len(test_x) != 0:
            test_ds = data.TensorDataset(test_x, test_y)
            test_dl = data.DataLoader(
                dataset=test_ds,
                batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
                shuffle=True,
                drop_last=False,
            )
        else:
            test_dl = None

        train_h5.close()
        test_h5.close()
        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        return [
            "n_class",
        ] + super().extra_repr_keys()

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "cnn_cifar": mnn.CNNCifar(num_classes=self.n_class),
            "resnet10": mnn.ResNet10(num_classes=self.n_class),
        }

    @property
    def doi(self) -> str:
        return None  # TODO

    @property
    def label_map(self) -> dict:
        return {
            10: CIFAR10_LABEL_MAP,
            100: CIFAR100_FINE_LABEL_MAP,
        }[self.n_class]

    def view_image(self, client_idx: int, image_idx: int) -> None:
        import matplotlib.pyplot as plt

        if client_idx >= len(self._client_ids_train):
            raise ValueError(
                f"client_idx should be less than {len(self._client_ids_train)}"
            )
        client_id = self._client_ids_train[client_idx]

        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")

        tot_img = train_h5[self._EXAMPLE][client_id][self._IMGAE][()]
        tot_label = train_h5[self._EXAMPLE][client_id][self._LABEL][()]
        if client_id in self._client_ids_test:
            tot_img = np.vstack(
                [
                    tot_img,
                    test_h5[self._EXAMPLE][client_id][self._IMGAE][()],
                ]
            )
            tot_label = np.concatenate(
                [
                    tot_label,
                    test_h5[self._EXAMPLE][client_id][self._LABEL][()],
                ]
            )
        if image_idx >= len(tot_img):
            raise ValueError(f"image_idx should be less than {len(tot_img)}")

        train_h5.close()
        test_h5.close()

        img = tot_img[image_idx]
        label = tot_label[image_idx]
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.title(
            f"client_id: {client_id}, label: {label} ({self.lable_map[int(label)]})"
        )
        plt.show()


class FedCIFAR100(FedCIFAR):

    __name__ = "FedCIFAR100"

    def __init__(self, datadir: Optional[Union[str, Path]] = None) -> None:
        super().__init__(100, datadir)

    @property
    def url(self) -> str:
        return "https://fedml.s3-us-west-1.amazonaws.com/fed_cifar100.tar.bz2"


def _data_transforms_fed_cifar(
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    train: bool = True,
    crop_size: Sequence[int] = (24, 24),
) -> Callable:
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    if mean is None:
        mean = CIFAR_MEAN
    if std is None:
        std = CIFAR_STD
    if train:
        return transforms.Compose(
            [
                # transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                # transforms.CenterCrop(crop_size),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
