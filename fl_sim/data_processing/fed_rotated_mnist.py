import gzip
import posixpath
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Callable

import numpy as np
import requests
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

from ..utils.const import (
    CACHED_DATA_DIR,
    MNIST_LABEL_MAP,
    MNIST_MEAN,
    MNIST_STD,
)
from ..utils._download_data import http_get
from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset
from ._register import register_fed_dataset
from ._ops import (
    ImageArrayToTensor,
    CategoricalLabelToTensor,
    FixedDegreeRotation,
    distribute_images,
)


__all__ = [
    "FedRotatedMNIST",
]


@register_fed_dataset()
class FedRotatedMNIST(FedVisionDataset):
    """MNIST dataset with rotation augmentation.

    The rotations are fixed and are multiples of 360 / num_rotations [1]_.

    The original MNIST dataset [2]_ contains 60,000 training images and 10,000 test images.
    Images are 28x28 grayscale images in 10 classes (0-9 handwritten digits).

    Parameters
    ----------
    datadir : str or pathlib.Path, optional
        Path to store the dataset. If not specified, the default path is used.
    num_rotations : int, default 4
        Number of rotations to apply to the images in the dataset.
        Typical values are 2, 4.
    num_clients : int, default 2400
        Number of clients to simulate.
        Typical values are 1200, 2400, 4800.
    transform : str or callable, default 'none'
        Transform (augmentation) to apply to the dataset.
        If 'none', no augmentation is applied,
        only the normalization transform is applied.
    seed : int, default 0
        Random seed for reproducibility.

    References
    ----------
    .. [1] Ghosh, A., Chung, J., Yin, D., & Ramchandran, K. (2020).
           An efficient framework for clustered federated learning.
           Advances in Neural Information Processing Systems, 33, 19586-19597.
    .. [2] https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html#MNIST

    """

    __name__ = "FedRotatedMNIST"

    def __init__(
        self,
        datadir: Optional[Union[Path, str]] = None,
        num_rotations: int = 4,
        num_clients: int = 2400,
        transform: Optional[Union[str, Callable]] = "none",
        seed: int = 0,
    ) -> None:
        self.num_rotations = num_rotations
        self.num_clients = num_clients
        assert self.num_clients % self.num_rotations == 0
        super().__init__(datadir=datadir, transform=transform, seed=seed)

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
        default_datadir = CACHED_DATA_DIR / "fed-rotated-mnist"
        self.datadir = Path(datadir or default_datadir).expanduser().resolve()
        self.datadir.mkdir(parents=True, exist_ok=True)

        # download if needed
        self.download_if_needed()

        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TEST_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TRAIN_FILE = {
            "images": self.url["train-images"],
            "labels": self.url["train-labels"],
        }
        self.DEFAULT_TEST_FILE = {
            "images": self.url["test-images"],
            "labels": self.url["test-labels"],
        }
        self._IMGAE = "image"
        self._LABEL = "label"

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # set transforms for creating dataset
        self.transform = transforms.Compose(
            [
                ImageArrayToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        )
        self.target_transform = CategoricalLabelToTensor()

        # load data
        self._train_data_dict = {}
        self._test_data_dict = {}
        for key, fn in self.url.items():
            with gzip.open(self.datadir / fn, "rb") as f:
                part, name = key.split("-")
                if name == "images":
                    data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                        -1, 28, 28
                    )
                    name = self._IMGAE
                else:  # name == "labels"
                    data = np.frombuffer(f.read(), np.uint8, offset=8)
                    name = self._LABEL
                if part == "train":
                    self._train_data_dict[name] = data
                else:  # part == "test"
                    self._test_data_dict[name] = data

        original_num_images = {
            "train": len(self._train_data_dict[self._LABEL]),
            "test": len(self._test_data_dict[self._LABEL]),
        }

        # set n_class
        self._n_class = len(
            np.unique(
                np.concatenate(
                    [
                        self._train_data_dict[self._LABEL],
                        self._test_data_dict[self._LABEL],
                    ]
                )
            )
        )

        # distribute data to clients
        self.indices = {}
        self.indices["train"] = distribute_images(
            original_num_images["train"],
            self.num_clients // self.num_rotations,
            random=True,
        )
        self.indices["test"] = distribute_images(
            original_num_images["test"],
            self.num_clients // self.num_rotations,
            random=False,
        )

        # perform rotation, and distribute data to clients
        print("Performing rotation...")
        angles = np.arange(0, 360, 360 / self.num_rotations)[1:]
        raw_images = {
            "train": torch.from_numpy(self._train_data_dict[self._IMGAE].copy()),
            "test": torch.from_numpy(self._test_data_dict[self._IMGAE].copy()),
        }
        raw_labels = {
            "train": self._train_data_dict[self._LABEL].copy(),
            "test": self._test_data_dict[self._LABEL].copy(),
        }
        for idx, angle in enumerate(angles):
            transform = FixedDegreeRotation(angle)
            self._train_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._train_data_dict[self._IMGAE],
                    transform(raw_images["train"]).numpy(),
                ]
            )
            self._train_data_dict[self._LABEL] = np.concatenate(
                [
                    self._train_data_dict[self._LABEL],
                    raw_labels["train"].copy(),
                ]
            )
            self._test_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._test_data_dict[self._IMGAE],
                    transform(raw_images["test"]).numpy(),
                ]
            )
            self._test_data_dict[self._LABEL] = np.concatenate(
                [
                    self._test_data_dict[self._LABEL],
                    raw_labels["test"].copy(),
                ]
            )
            self.indices["train"].extend(
                distribute_images(
                    np.arange(original_num_images["train"])
                    + (idx + 1) * original_num_images["train"],
                    self.num_clients // self.num_rotations,
                    random=True,
                )
            )
            self.indices["test"].extend(
                distribute_images(
                    np.arange(original_num_images["test"])
                    + (idx + 1) * original_num_images["test"],
                    self.num_clients // self.num_rotations,
                    random=False,
                )
            )
        del raw_images, raw_labels

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
        if client_idx is None:
            train_slice = slice(None)
            test_slice = slice(None)
        else:
            train_slice = self.indices["train"][client_idx]
            test_slice = self.indices["test"][client_idx]

        train_ds = torchdata.TensorDataset(
            self.transform(
                self._train_data_dict[self._IMGAE][train_slice].copy()
            ).unsqueeze(1),
            self.target_transform(
                self._train_data_dict[self._LABEL][train_slice].copy()
            ),
        )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            self.transform(
                self._test_data_dict[self._IMGAE][test_slice].copy()
            ).unsqueeze(1),
            self.target_transform(self._test_data_dict[self._LABEL][test_slice].copy()),
        )
        test_dl = torchdata.DataLoader(
            dataset=test_ds,
            batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        return [
            "n_class",
        ] + super().extra_repr_keys()

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
    def mirror(self) -> Dict[str, str]:
        """Mirror sites for downloading the dataset."""
        return {
            "lecun": "http://yann.lecun.com/exdb/mnist/",
            "aws": "https://ossci-datasets.s3.amazonaws.com/mnist/",
        }

    @property
    def url(self) -> Dict[str, str]:
        """URLs for downloading the dataset."""
        return {
            "train-images": "train-images-idx3-ubyte.gz",
            "train-labels": "train-labels-idx1-ubyte.gz",
            "test-images": "t10k-images-idx3-ubyte.gz",
            "test-labels": "t10k-labels-idx1-ubyte.gz",
        }

    def download_if_needed(self) -> None:
        """Download data if needed."""
        default_mirror = "lecun"
        alt_mirror = [k for k in self.mirror if k != default_mirror][0]
        # check if default_mirror is available
        if requests.get(self.mirror[default_mirror]).status_code == 200:
            base_url = self.mirror[default_mirror]
        else:
            base_url = self.mirror[alt_mirror]
        for key, fn in self.url.items():
            url = posixpath.join(base_url, fn)
            local_fn = self.datadir / fn
            if local_fn.exists():
                print(f"{key} exists, skip downloading")
                continue
            http_get(url, self.datadir, extract=False)

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """A set of candidate models."""
        return {
            "cnn_mnist": mnn.CNNMnist(num_classes=self.n_class),
            "cnn_femmist_tiny": mnn.CNNFEMnist_Tiny(num_classes=self.n_class),
            "cnn_femmist": mnn.CNNFEMnist(num_classes=self.n_class),
            # "resnet10": mnn.ResNet10(num_classes=self.n_class),
            "mlp": mnn.MLP(dim_in=28 * 28, dim_out=self.n_class, ndim=2),
        }

    @property
    def doi(self) -> List[str]:
        """DOI(s) related to the dataset."""
        # TODO: add doi of MNIST and IFCA
        return None

    @property
    def label_map(self) -> dict:
        """Label map for the dataset."""
        return MNIST_LABEL_MAP

    def view_image(self, client_idx: int, image_idx: int) -> None:
        """View a single image.

        Parameters
        ----------
        client_idx : int
            Index of the client on which the image is located.
        image_idx : int
            Index of the image in the client.

        Returns
        -------
        None

        """
        import matplotlib.pyplot as plt

        if client_idx >= self.num_clients:
            raise ValueError(
                f"client_idx must be less than {self.num_clients}, got {client_idx}"
            )

        total_num_images = len(self.indices["train"][client_idx]) + len(
            self.indices["test"][client_idx]
        )
        if image_idx >= total_num_images:
            raise ValueError(
                f"image_idx must be less than {total_num_images}, got {image_idx}"
            )
        if image_idx < len(self.indices["train"][client_idx]):
            image = self._train_data_dict[self._IMGAE][
                self.indices["train"][client_idx][image_idx]
            ]
            label = self._train_data_dict[self._LABEL][
                self.indices["train"][client_idx][image_idx]
            ]
            image_idx = self.indices["train"][client_idx][image_idx]
            angle = (
                image_idx
                // (len(self._train_data_dict[self._IMGAE]) // self.num_rotations)
                * (360 // self.num_rotations)
            )
        else:
            image_idx -= len(self.indices["train"][client_idx])
            image = self._test_data_dict[self._IMGAE][
                self.indices["test"][client_idx][image_idx]
            ]
            label = self._test_data_dict[self._LABEL][
                self.indices["test"][client_idx][image_idx]
            ]
            image_idx = self.indices["test"][client_idx][image_idx]
            angle = (
                image_idx
                // (len(self._test_data_dict[self._IMGAE]) // self.num_rotations)
                * (360 // self.num_rotations)
            )
        plt.imshow(image, cmap="gray")
        plt.title(
            f"image_idx: {image_idx}, label: {label} ({self.label_map[int(label)]}), "
            f"angle: {angle}"
        )
        plt.show()

    def random_grid_view(
        self, nrow: int, ncol: int, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Select randomly `nrow` x `ncol` images from the dataset
        and plot them in a grid.

        Parameters
        ----------
        nrow : int
            Number of rows in the grid.
        ncol : int
            Number of columns in the grid.
        save_path : Union[str, Path], optional
            Path to save the figure. If ``None``, do not save the figure.

        Returns
        -------
        None

        """
        import matplotlib.pyplot as plt

        rng = np.random.default_rng()

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1, nrow * 1))
        selected = []
        for i in range(nrow):
            for j in range(ncol):
                while True:
                    client_idx = rng.integers(self.num_clients)
                    image_idx = rng.integers(len(self.indices["train"][client_idx]))
                    if (client_idx, image_idx) not in selected:
                        selected.append((client_idx, image_idx))
                        break
                image = self._train_data_dict[self._IMGAE][
                    self.indices["train"][client_idx][image_idx]
                ]
                axes[i, j].imshow(image, cmap="gray")
                axes[i, j].axis("off")
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.tight_layout()
        plt.show()
