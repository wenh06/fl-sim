"""
federeated EMNIST
"""

import warnings
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import h5py
import numpy as np
import torch
import torch.utils.data as torchdata

from ..utils.const import CACHED_DATA_DIR, EMNIST_LABEL_MAP
from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset


__all__ = [
    "FedEMNIST",
]


FED_EMNIST_DATA_DIR = CACHED_DATA_DIR / "fed_emnist"
FED_EMNIST_DATA_DIR.mkdir(parents=True, exist_ok=True)


class FedEMNIST(FedVisionDataset):
    """
    Federated EMNIST dataset extends MNIST dataset with upper and lower case English characters
    Data partition is the same as TensorFlow Federated (TFF) (ref. [1]_) with the following statistics:

    +-----------+---------------+----------------+--------------+---------------+
    | DATASET   | TRAIN CLIENTS | TRAIN EXAMPLES | TEST CLIENTS | TEST EXAMPLES |
    +===========+===============+================+==============+===============+
    | EMNIST-62 | 3,400         | 671,585        | 3,400        | 77,483        |
    +-----------+---------------+----------------+--------------+---------------+

    Most methods in this class are modified from FedML (ref. [2]_).

    NOTE: the images are processed using min-max normalization to range [0, 1].

    Parameters
    ----------
    datadir : Union[Path, str], optional
        Directory to store data.
        If ``None``, use default directory.
    transform : Union[str, Callable], default "none"
        Transform to apply to data. Conventions:
        ``"none"`` means no transform, using TensorDataset.
    seed : int, default 0
        Random seed for data partitioning.
    **extra_config : dict, optional
        Extra configurations.

    References
    ----------
    .. [1] https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist
    .. [2] https://github.com/FedML-AI/FedML/tree/master/python/fedml/data/FederatedEMNIST

    """

    __name__ = "FedEMNIST"

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        self.datadir = Path(datadir or FED_EMNIST_DATA_DIR).expanduser().resolve()
        self.datadir.mkdir(parents=True, exist_ok=True)

        self.DEFAULT_TRAIN_CLIENTS_NUM = 3400
        self.DEFAULT_TEST_CLIENTS_NUM = 3400
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = "fed_emnist_train.h5"
        self.DEFAULT_TEST_FILE = "fed_emnist_test.h5"
        self._IMGAE = "pixels"

        self.criterion = torch.nn.CrossEntropyLoss()

        if self.transform != "none":
            warnings.warn(
                "The images are not raw pixels, but processed. "
                "The transform argument will be ignored.",
                RuntimeWarning,
            )
            self.transform = "none"

        self.download_if_needed()

        # client id list
        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        with h5py.File(str(train_file_path), "r") as train_h5, h5py.File(
            str(test_file_path), "r"
        ) as test_h5:
            self._client_ids_train = list(train_h5[self._EXAMPLE].keys())
            self._client_ids_test = list(test_h5[self._EXAMPLE].keys())
            self._n_class = len(
                np.unique(
                    [
                        train_h5[self._EXAMPLE][self._client_ids_train[idx]][
                            self._LABEL
                        ][0]
                        for idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM)
                    ]
                )
            )

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")
        train_x, train_y, test_x, test_y = [], [], [], []

        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self._client_ids_train
            test_ids = self._client_ids_test
        else:
            # get ids of single client
            train_ids = [self._client_ids_train[client_idx]]
            test_ids = [self._client_ids_test[client_idx]]

        # load data in numpy format from h5 file
        train_x = np.vstack(
            [
                train_h5[self._EXAMPLE][client_id][self._IMGAE][()]
                for client_id in train_ids
            ]
        )
        train_y = np.concatenate(
            [
                train_h5[self._EXAMPLE][client_id][self._LABEL][()]
                for client_id in train_ids
            ]
        )
        test_x = np.vstack(
            [
                test_h5[self._EXAMPLE][client_id][self._IMGAE][()]
                for client_id in test_ids
            ]
        )
        test_y = np.concatenate(
            [
                test_h5[self._EXAMPLE][client_id][self._LABEL][()]
                for client_id in test_ids
            ]
        )

        # dataloader
        train_ds = torchdata.TensorDataset(
            torch.from_numpy(train_x).unsqueeze(1),
            torch.from_numpy(train_y.astype(np.int64)),
        )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            torch.from_numpy(test_x).unsqueeze(1),
            torch.from_numpy(test_y.astype(np.int64)),
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
    def url(self) -> str:
        return "https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "cnn_femmist_tiny": mnn.CNNFEMnist_Tiny(),
            "cnn_femmist": mnn.CNNFEMnist(),
            # "resnet10": mnn.ResNet10(num_classes=self.n_class),
            "mlp": mnn.MLP(dim_in=28 * 28, dim_out=self.n_class, ndim=2),
        }

    @property
    def doi(self) -> List[str]:
        return [
            "10.1109/5.726791",  # MNIST
            "10.1109/ijcnn.2017.7966217",  # EMNIST
            "10.48550/ARXIV.1812.01097",  # LEAF
        ]

    @property
    def label_map(self) -> dict:
        return EMNIST_LABEL_MAP

    def view_image(self, client_idx: int, image_idx: int) -> None:
        import matplotlib.pyplot as plt

        if client_idx >= len(self._client_ids_train):
            raise ValueError(
                f"client_idx must be less than {len(self._client_ids_train)}"
            )
        client_id = self._client_ids_train[client_idx]

        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")

        tot_img = np.vstack(
            [
                train_h5[self._EXAMPLE][client_id][self._IMGAE][()],
                test_h5[self._EXAMPLE][client_id][self._IMGAE][()],
            ]
        )
        tot_label = np.concatenate(
            [
                train_h5[self._EXAMPLE][client_id][self._LABEL][()],
                test_h5[self._EXAMPLE][client_id][self._LABEL][()],
            ]
        )
        if image_idx >= len(tot_img):
            raise ValueError(f"image_idx should be less than {len(tot_img)}")

        train_h5.close()
        test_h5.close()

        img = (tot_img[image_idx] * 255).astype(np.uint8)
        label = tot_label[image_idx]
        plt.imshow(img, cmap="gray")
        plt.title(
            f"client_id: {client_id}, label: {label} ({self.label_map[int(label)]})"
        )
        plt.show()
