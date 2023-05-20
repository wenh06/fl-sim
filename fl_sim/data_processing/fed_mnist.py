"""
federeated MNIST
"""

import json
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
import torch
import torch.utils.data as torchdata

from ..utils.const import CACHED_DATA_DIR, MNIST_LABEL_MAP
from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset


__all__ = [
    "FedMNIST",
]


FED_MNIST_DATA_DIR = CACHED_DATA_DIR / "fed_mnist"
FED_MNIST_DATA_DIR.mkdir(parents=True, exist_ok=True)


class FedMNIST(FedVisionDataset):
    """MNIST is a dataset to study image classification of handwritten digits 0-9.

    To simulate a heterogeneous setting, FedML distribute the data
    among 1000 devices such that each device has samples of only 2 digits
    and the number of samples per device follows a power law.

    References
    ----------
    1. https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/MNIST/
    2. Federated Optimization in Heterogeneous Networks (https://arxiv.org/pdf/1812.06127.pdf). MLSys 2020.

    """

    __name__ = "FedMNIST"

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        self.datadir = Path(datadir or FED_MNIST_DATA_DIR).expanduser().resolve()

        self.DEFAULT_TRAIN_CLIENTS_NUM = 1000
        self.DEFAULT_TEST_CLIENTS_NUM = 1000
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = "train/all_data_0_niid_0_keep_10_train_9.json"
        self.DEFAULT_TEST_FILE = "test/all_data_0_niid_0_keep_10_test_9.json"
        self._EXAMPLE = "user_data"
        self._IMGAE = "x"
        self._LABEL = "y"

        self.criterion = torch.nn.CrossEntropyLoss()

        self.download_if_needed()

        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        self._train_data_dict = json.loads(train_file_path.read_text())
        self._test_data_dict = json.loads(test_file_path.read_text())
        self._client_ids_train = self._train_data_dict["users"]
        self._client_ids_test = self._test_data_dict["users"]

        self._n_class = len(
            np.unique(
                np.concatenate(
                    [
                        self._train_data_dict[self._EXAMPLE][
                            self._client_ids_train[idx]
                        ][self._LABEL]
                        for idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM)
                    ]
                )
            )
        )

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

        # load data
        train_x = np.vstack(
            [
                self._train_data_dict[self._EXAMPLE][client_id][self._IMGAE]
                for client_id in train_ids
            ]
        )
        train_y = np.concatenate(
            [
                self._train_data_dict[self._EXAMPLE][client_id][self._LABEL]
                for client_id in train_ids
            ]
        )
        test_x = np.vstack(
            [
                self._test_data_dict[self._EXAMPLE][client_id][self._IMGAE]
                for client_id in test_ids
            ]
        )
        test_y = np.concatenate(
            [
                self._test_data_dict[self._EXAMPLE][client_id][self._LABEL]
                for client_id in test_ids
            ]
        )

        # dataloader
        train_ds = torchdata.TensorDataset(
            torch.from_numpy(
                train_x.reshape((-1, 28, 28)).astype(np.float32)
            ).unsqueeze(1),
            torch.from_numpy(train_y.astype(np.int64)),
        )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            torch.from_numpy(test_x.reshape((-1, 28, 28)).astype(np.float32)).unsqueeze(
                1
            ),
            torch.from_numpy(test_y.astype(np.int64)),
        )
        test_dl = torchdata.DataLoader(
            dataset=test_ds,
            batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

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
        return "https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "cnn_mnist": mnn.CNNMnist(num_classes=self.n_class),
            "cnn_femmist_tiny": mnn.CNNFEMnist_Tiny(num_classes=self.n_class),
            "cnn_femmist": mnn.CNNFEMnist(num_classes=self.n_class),
            # "resnet10": mnn.ResNet10(num_classes=self.n_class),
            "mlp": mnn.MLP(dim_in=28 * 28, dim_out=self.n_class, ndim=2),
        }

    @property
    def doi(self) -> List[str]:
        # TODO: add FedML doi and MNIST doi
        return ["10.48550/ARXIV.1812.06127"]

    @property
    def label_map(self) -> dict:
        return MNIST_LABEL_MAP

    def view_image(self, client_idx: int, image_idx: int) -> None:
        import matplotlib.pyplot as plt

        if client_idx >= len(self._train_data_dict["users"]):
            raise ValueError(
                f"client_idx must be less than {len(self._train_data_dict['users'])}"
            )
        client_id = self._train_data_dict["users"][client_idx]
        total_num_images = len(
            self._train_data_dict["user_data"][client_id]["x"]
        ) + len(self._test_data_dict["user_data"][client_id]["x"])
        if image_idx >= total_num_images:
            raise ValueError(
                f"image_idx must be less than {total_num_images} (total number of images)"
            )
        if image_idx < len(self._train_data_dict["user_data"][client_id]["x"]):
            image = np.array(self._train_data_dict["user_data"][client_id]["x"])[
                image_idx
            ].reshape(28, 28)
            image = (
                255 - (image - image.min()) / (image.max() - image.min()) * 255
            ).astype(np.uint8)
            label = self._train_data_dict["user_data"][client_id]["y"][image_idx]
        else:
            image_idx -= len(self._train_data_dict["user_data"][client_id]["x"])
            image = np.array(self._test_data_dict["user_data"][client_id]["x"])[
                image_idx
            ].reshape(28, 28)
            image = (
                255 - (image - image.min()) / (image.max() - image.min()) * 255
            ).astype(np.uint8)
            label = self._train_data_dict["user_data"][client_id]["y"][image_idx]
        plt.imshow(image, cmap="gray")
        plt.title(
            f"client_id: {client_id}, label: {label} ({self.label_map[int(label)]})"
        )
        plt.show()
