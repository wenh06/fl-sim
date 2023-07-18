"""
"""

import warnings
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
from scipy.io import loadmat
import torch
import torch.utils.data as torchdata

from ..utils.const import CACHED_DATA_DIR, MNIST_LABEL_MAP
from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset
from ._register import register_fed_dataset


__all__ = [
    "FedProxMNIST",
]


FEDPROX_MNIST_DATA_DIR = CACHED_DATA_DIR / "fedprox_mnist"
FEDPROX_MNIST_DATA_DIR.mkdir(parents=True, exist_ok=True)


@register_fed_dataset()
class FedProxMNIST(FedVisionDataset):
    """Federeated MNIST used in the FedProx paper [1]_, [2]_,
    where the data is partitioned in a non-IID manner.

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
    .. [1] https://github.com/litian96/FedProx/tree/master/data/mnist
    .. [2] https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py

    """

    __name__ = "FedProxMNIST"

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        self.datadir = Path(datadir or FEDPROX_MNIST_DATA_DIR).expanduser().resolve()

        if hasattr(self, "num_clients"):
            self.DEFAULT_TRAIN_CLIENTS_NUM = self.num_clients
            self.DEFAULT_TEST_CLIENTS_NUM = self.num_clients
        else:
            self.DEFAULT_TRAIN_CLIENTS_NUM = 1000
            self.DEFAULT_TEST_CLIENTS_NUM = 1000
        self.DEFAULT_BATCH_SIZE = 20

        self.DEFAULT_TRAIN_FILE = "fedprox-mnist.mat"
        self.DEFAULT_TEST_FILE = "fedprox-mnist.mat"

        self._EXAMPLE = ""
        self._IMGAE = "data"
        self._LABEL = "label"

        if self.transform != "none":
            warnings.warn(
                "The images are not raw pixels, but processed. "
                "The transform argument will be ignored.",
                RuntimeWarning,
            )
            self.transform = "none"

        self.criterion = torch.nn.CrossEntropyLoss()

        self.download_if_needed()
        self.__raw_data = loadmat(self.datadir / self.DEFAULT_TRAIN_FILE)
        self._client_data = generate_niid(
            self.__raw_data, num_clients=self.DEFAULT_TRAIN_CLIENTS_NUM, seed=self.seed
        )

        self._client_ids_train = list(range(self.DEFAULT_TRAIN_CLIENTS_NUM))
        self._client_ids_test = list(range(self.DEFAULT_TEST_CLIENTS_NUM))

        self._n_class = len(
            np.unique(
                np.concatenate(
                    [
                        item["train_y"].tolist() + item["test_y"].tolist()
                        for item in self._client_data
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
            [self._client_data[client_id]["train_x"] for client_id in train_ids]
        )
        train_y = np.concatenate(
            [self._client_data[client_id]["train_y"] for client_id in train_ids]
        )
        test_x = np.vstack(
            [self._client_data[client_id]["test_x"] for client_id in test_ids]
        )
        test_y = np.concatenate(
            [self._client_data[client_id]["test_y"] for client_id in test_ids]
        )

        # dataloader
        train_ds = torchdata.TensorDataset(
            torch.from_numpy(train_x.astype(np.float32)).unsqueeze(1),
            torch.from_numpy(train_y.astype(np.int64)),
        )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            torch.from_numpy(test_x.astype(np.float32)).unsqueeze(1),
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
        # https://drive.google.com/file/d/1tCEcJgRJ8NdRo11UJZR6WSKMNdmox4GC/view?usp=sharing
        # "http://218.245.5.12/NLP/federated/fedprox-mnist.zip"
        return "https://www.dropbox.com/s/ndri55jt0w9juk1/fedprox-mnist.zip?dl=1"

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
        return [
            "10.1109/5.726791",  # MNIST
            "10.48550/ARXIV.1812.01097",  # LEAF
            "10.48550/ARXIV.1812.06127",  # FedProx
        ]

    @property
    def raw_data(self) -> Dict[str, np.ndarray]:
        return self.__raw_data

    @property
    def label_map(self) -> dict:
        return MNIST_LABEL_MAP

    def view_image(self, client_idx: int, image_idx: int) -> None:
        import matplotlib.pyplot as plt

        if client_idx >= self.DEFAULT_TRAIN_CLIENTS_NUM:
            raise ValueError(
                f"client_idx should be less than {self.DEFAULT_TRAIN_CLIENTS_NUM}"
            )
        tot_images = (
            self._client_data[client_idx]["train_x"].shape[0]
            + self._client_data[client_idx]["test_x"].shape[0]
        )
        if image_idx >= tot_images:
            raise ValueError(f"image_idx should be less than {tot_images}")
        if image_idx < self._client_data[client_idx]["train_x"].shape[0]:
            img = self._client_data[client_idx]["train_x"][image_idx]
            label = self._client_data[client_idx]["train_y"][image_idx]
        else:
            img = self._client_data[client_idx]["test_x"][
                image_idx - self._client_data[client_idx]["train_x"].shape[0]
            ]
            label = self._client_data[client_idx]["test_y"][
                image_idx - self._client_data[client_idx]["train_x"].shape[0]
            ]

        img = img + img.min()
        # to 0-255
        img = (img * 255 / img.max()).astype(np.uint8)
        plt.imshow(img, cmap="gray")
        plt.title(f"client {client_idx}, label {label} ({self.label_map[int(label)]})")
        plt.show()

    def random_grid_view(
        self, nrow: int, ncol: int, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Select randomly `nrow` x `ncol` images from the dataset
        and plot them in a grid.
        """
        import matplotlib.pyplot as plt

        rng = np.random.default_rng()

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1, nrow * 1))
        selected = []
        for i in range(nrow):
            for j in range(ncol):
                while True:
                    client_idx = rng.integers(self.DEFAULT_TRAIN_CLIENTS_NUM)
                    tot_images = (
                        self._client_data[client_idx]["train_x"].shape[0]
                        + self._client_data[client_idx]["test_x"].shape[0]
                    )
                    image_idx = rng.integers(tot_images)
                    if (client_idx, image_idx) not in selected:
                        selected.append((client_idx, image_idx))
                        break
                if image_idx < self._client_data[client_idx]["train_x"].shape[0]:
                    img = self._client_data[client_idx]["train_x"][image_idx]
                    label = self._client_data[client_idx]["train_y"][image_idx]
                else:
                    img = self._client_data[client_idx]["test_x"][
                        image_idx - self._client_data[client_idx]["train_x"].shape[0]
                    ]

                img = img + img.min()
                # to 0-255
                img = (img * 255 / img.max()).astype(np.uint8)
                axes[i, j].imshow(img, cmap="gray")
                axes[i, j].axis("off")
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.tight_layout()
        plt.show()


def generate_niid(
    mnist_data: Dict[str, np.ndarray],
    num_clients: int = 1000,
    lower_bound: int = 10,
    class_per_client: int = 2,
    seed: int = 42,
    train_ratio: float = 0.9,
) -> List[Dict[str, np.ndarray]]:
    """
    modified from
    https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py
    """
    NUM_CLASSES = 10
    IMG_SHAPE = (28, 28)
    mnist_data["data"] = (mnist_data["data"] / 255.0).astype(np.float32)
    eps = 1e-5
    options = dict(axis=0, keepdims=True)
    mean = mnist_data["data"].mean(**options)
    std = mnist_data["data"].std(**options)
    mnist_data["data"] = (mnist_data["data"] - mean) / (std + eps)
    mnist_data["data"] = mnist_data["data"].T.reshape((-1, *IMG_SHAPE))
    mnist_data["label"] = mnist_data["label"].flatten()

    class_inds = {i: np.where(mnist_data["label"] == i)[0] for i in range(NUM_CLASSES)}
    class_nums = [lower_bound // class_per_client for _ in range(class_per_client - 1)]
    class_nums.append(lower_bound - sum(class_nums))

    clients_data = [
        {
            k: np.empty((0, *IMG_SHAPE), dtype=np.float32)
            if k.startswith("train")
            else np.array([], dtype=np.int64)
            for k in [
                "train_x",
                "train_y",
                "test_x",
                "test_y",
            ]
        }
        for _ in range(num_clients)
    ]
    # idx = np.zeros(NUM_CLASSES, dtype=np.int64)
    idx = {i: 0 for i in range(NUM_CLASSES)}
    for c in range(num_clients):
        for j, n in enumerate(class_nums):
            label = (c + j) % NUM_CLASSES
            inds = class_inds[label][idx[label] : idx[label] + n]
            clients_data[c]["train_x"] = np.append(
                clients_data[c]["train_x"], mnist_data["data"][inds, ...], axis=0
            )
            clients_data[c]["train_y"] = np.append(
                clients_data[c]["train_y"], np.full_like(inds, label, dtype=np.int64)
            )
            idx[label] += n
    # print(f"idx = {idx}")
    # print(f"class_inds = {[(l, len(class_inds[l])) for l in range(NUM_CLASSES)]}")

    rng = np.random.default_rng(seed)
    probs = rng.lognormal(
        0, 2.0, (NUM_CLASSES, num_clients // NUM_CLASSES, class_per_client)
    )
    probs = (
        np.array([[[len(class_inds[i]) - idx[i]]] for i in range(NUM_CLASSES)])
        * probs
        / probs.sum(axis=(1, 2), keepdims=True)
    )
    for c in range(num_clients):
        for j, n in enumerate(class_nums):
            label = (c + j) % NUM_CLASSES
            num_samples = round(probs[label, c // NUM_CLASSES, j])
            if idx[label] + num_samples < len(class_inds[label]):
                inds = class_inds[label][idx[label] : idx[label] + num_samples]
                clients_data[c]["train_x"] = np.append(
                    clients_data[c]["train_x"], mnist_data["data"][inds, ...], axis=0
                )
                clients_data[c]["train_y"] = np.append(
                    clients_data[c]["train_y"],
                    np.full_like(inds, label, dtype=np.int64),
                )
                idx[label] += num_samples
        num_samples = clients_data[c]["train_x"].shape[0]
        inds = rng.choice(num_samples, num_samples, replace=False)
        train_len = int(train_ratio * num_samples)
        clients_data[c]["test_x"] = clients_data[c]["train_x"][inds[train_len:], ...]
        clients_data[c]["test_y"] = clients_data[c]["train_y"][inds[train_len:]]
        clients_data[c]["train_x"] = clients_data[c]["train_x"][inds[:train_len], ...]
        clients_data[c]["train_y"] = clients_data[c]["train_y"][inds[:train_len]]
    # print(f"idx = {idx}")

    return clients_data
