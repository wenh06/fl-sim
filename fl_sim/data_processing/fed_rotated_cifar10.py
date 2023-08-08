import pickle
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Callable

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

from ..utils.const import (
    CACHED_DATA_DIR,
    CIFAR10_LABEL_MAP,
    CIFAR10_MEAN,
    CIFAR10_STD,
)
from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset, VisionDataset
from ._register import register_fed_dataset
from ._ops import (
    ImageArrayToTensor,
    CategoricalLabelToTensor,
    FixedDegreeRotation,
    ImageTensorScale,
    distribute_images,
)


__all__ = [
    "FedRotatedCIFAR10",
]


@register_fed_dataset()
class FedRotatedCIFAR10(FedVisionDataset):
    """CIFAR10 dataset with rotation augmentation.

    The rotations are fixed and are multiples of 360 / num_rotations [1]_

    The original CIFAR10 dataset [2]_ contains 50k training images and 10k test images.
    Images are 32x32 RGB images in 10 classes.

    Parameters
    ----------
    datadir : str or pathlib.Path, optional
        Path to store the dataset. If not specified, the default path is used.
    num_rotations : int, default 2
        Number of rotations to apply to the images in the dataset.
    num_clients : int, default 200
        Number of clients to simulate.
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
    .. [2] https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10

    """

    __name__ = "FedRotatedCIFAR10"

    def __init__(
        self,
        datadir: Optional[Union[Path, str]] = None,
        num_rotations: int = 2,
        num_clients: int = 200,
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
        default_datadir = CACHED_DATA_DIR / "fed-rotated-cifar10"
        self.datadir = Path(datadir or default_datadir).expanduser().resolve()
        self.datadir.mkdir(parents=True, exist_ok=True)

        # download data
        self.download_if_needed()

        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TEST_CLIENTS_NUM = self.num_clients

        self.DEFAULT_TRAIN_FILE = [
            f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)
        ]
        self.DEFAULT_TEST_FILE = ["cifar-10-batches-py/test_batch"]
        self._IMGAE = "image"
        self._LABEL = "label"

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # set transforms for creating dataset
        if self.transform is None:
            # set dynamic transform for train set
            self.transform = transforms.Compose(
                [
                    transforms.AutoAugment(
                        policy=transforms.AutoAugmentPolicy.CIFAR10,
                    ),
                    ImageTensorScale(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
        self.target_transform = transforms.Compose([CategoricalLabelToTensor()])

        # load data
        self._train_data_dict = {
            self._IMGAE: np.empty((0, 3, 32, 32), dtype=np.uint8),
            self._LABEL: np.empty((0,), dtype=np.int64),
        }
        self._test_data_dict = {
            self._IMGAE: np.empty((0, 3, 32, 32), dtype=np.uint8),
            self._LABEL: np.empty((0,), dtype=np.int64),
        }

        for file in self.DEFAULT_TRAIN_FILE:
            data = pickle.loads((self.datadir / file).read_bytes(), encoding="bytes")
            self._train_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._train_data_dict[self._IMGAE],
                    data[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8),
                ]
            )
            self._train_data_dict[self._LABEL] = np.concatenate(
                [
                    self._train_data_dict[self._LABEL],
                    np.array(data[b"labels"]).astype(np.int64),
                ]
            )
        data = pickle.loads(
            (self.datadir / self.DEFAULT_TEST_FILE[0]).read_bytes(),
            encoding="bytes",
        )
        self._test_data_dict[self._IMGAE] = (
            data[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
        )
        self._test_data_dict[self._LABEL] = np.array(data[b"labels"]).astype(np.int64)

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

        # static transform
        static_transform = transforms.Compose(
            [
                ImageArrayToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        if self.transform == "none":
            # apply only static transform
            train_ds = torchdata.TensorDataset(
                static_transform(
                    self._train_data_dict[self._IMGAE][train_slice].copy()
                ),
                self.target_transform(
                    self._train_data_dict[self._LABEL][train_slice].copy()
                ),
            )
        else:
            # use non-trivial dynamic transform
            train_ds = VisionDataset(
                images=torch.from_numpy(
                    self._train_data_dict[self._IMGAE][train_slice].copy()
                ).to(torch.uint8),
                targets=self.target_transform(
                    self._train_data_dict[self._LABEL][train_slice].copy()
                ),
                transform=self.transform,
            )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            static_transform(self._test_data_dict[self._IMGAE][test_slice].copy()),
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
    def url(self) -> str:
        """URL for downloading the dataset."""
        return "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """A set of candidate models."""
        return {
            "cnn_cifar": mnn.CNNCifar(num_classes=self.n_class),
            "cnn_cifar_small": mnn.CNNCifar_Small(num_classes=self.n_class),
            "cnn_cifar_tiny": mnn.CNNCifar_Tiny(num_classes=self.n_class),
            "resnet10": mnn.ResNet10(num_classes=self.n_class),
        }

    @property
    def doi(self) -> List[str]:
        """DOI(s) related to the dataset."""
        # TODO: add doi of CIFAR10 and IFCA
        return None

    @property
    def label_map(self) -> dict:
        """Label map for the dataset."""
        return CIFAR10_LABEL_MAP

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
        # image: channel first to channel last
        image = image.transpose(1, 2, 0)
        plt.imshow(image)
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
                axes[i, j].imshow(image.transpose(1, 2, 0))
                axes[i, j].axis("off")
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.tight_layout()
        plt.show()
