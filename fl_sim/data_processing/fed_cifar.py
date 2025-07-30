from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

from ..models import nn as mnn
from ..models.utils import top_n_accuracy
from ..utils.const import (
    CACHED_DATA_DIR,
    CIFAR10_LABEL_MAP,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_FINE_LABEL_MAP,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from ._register import register_fed_dataset
from .fed_dataset import FedVisionDataset, VisionDataset
from ._noniid_partition import non_iid_partition_with_dirichlet_distribution, record_data_stats

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
    """Federated CIFAR10/100 dataset.

    This dataset is loaded from TensorFlow Federated (TFF) cifar100 load_data API [1]_,
    and saved as h5py files. This dataset is pre-divided into 500 training clients
    containing 50,000 examples in total, and 100 testing clients containing 10,000
    examples in total.

    The images are saved in the channel last format, i.e.,
    ``N x H x W x C``, **NOT** the usual channel first format for PyTorch.
    A single image (and similarly for label and coarse_label) can be accessed by

    .. code-block:: python

        with h5py.File(path, "r") as f:
            images = f["examples"]["0"]["image"][0]

    where ``path`` is the path to the h5py file, "0" is the client id, and 0 is the
    index of the image in the client's dataset.

    Most methods in this class are adopted and modified from FedML [2]_.

    Parameters
    ----------
    n_class : {10, 100}, default 10
        Number of classes in the dataset.
        10 for CIFAR10, 100 for CIFAR100.
    datadir : str or pathlib.Path, default None
        Path to the dataset directory. Default: ``None``.
        If ``None``, will use built-in default directory.
    transform : str or callable, default "none"
        Transformation to apply to the images. Default: ``"none"``.
        If ``"none"``, only static normalization will be applied.
        If callable, will be used as ``transform`` argument for
        ``VisionDataset``.
        If ``None``, will use default dynamic augmentation transform.
    seed : int, default: 0
        Random seed for data shuffling.
    **extra_config : dict, optional
        Extra configurations for the dataset.

    References
    ----------
    .. [1] https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data
    .. [2] https://github.com/FedML-AI/FedML/tree/master/python/fedml/data/fed_cifar100

    """

    __name__ = "FedCIFAR"

    def __init__(
        self,
        n_class: int = 100,
        datadir: Optional[Union[str, Path]] = None,
        transform: Optional[Union[str, Callable]] = "none",
        seed: int = 0,
        **extra_config: Any,
    ) -> None:
        self._n_class = n_class
        assert self.n_class in [
            100,  # 10 not implemented
        ]
        datadir = Path(datadir or FED_CIFAR_DATA_DIRS[n_class]).expanduser().resolve()
        datadir.mkdir(parents=True, exist_ok=True)
        super().__init__(datadir=datadir, transform=transform, seed=seed, **extra_config)

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
        self.DEFAULT_TRAIN_CLIENTS_NUM = 500
        self.DEFAULT_TEST_CLIENTS_NUM = 100
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = f"fed_cifar{self.n_class}_train.h5"
        self.DEFAULT_TEST_FILE = f"fed_cifar{self.n_class}_test.h5"

        # group name defined by tff in h5 file
        self._EXAMPLE = "examples"
        self._IMGAE = "image"
        self._LABEL = "label"

        # set default transform from torchvision
        if self.n_class == 10 and self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.AutoAugment(
                        policy=transforms.AutoAugmentPolicy.CIFAR10,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
        elif self.n_class == 100 and self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandAugment(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                ]
            )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.download_if_needed()

        # client id list
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
        train_x, train_y, test_x, test_y = [], [], [], []

        # load data in numpy format from h5 file
        if client_idx is None:
            train_x = np.vstack([train_h5[self._EXAMPLE][client_id][self._IMGAE][()] for client_id in self._client_ids_train])
            train_y = np.concatenate(
                [train_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in self._client_ids_train]
            )
            test_x = np.vstack([test_h5[self._EXAMPLE][client_id][self._IMGAE][()] for client_id in self._client_ids_test])
            test_y = np.concatenate([test_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in self._client_ids_test])
            print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        else:
            client_id_train = self._client_ids_train[client_idx]
            train_x = np.vstack([train_h5[self._EXAMPLE][client_id_train][self._IMGAE][()]])
            train_y = np.concatenate([train_h5[self._EXAMPLE][client_id_train][self._LABEL][()]])
            if client_idx <= len(self._client_ids_test) - 1:
                client_id_test = self._client_ids_test[client_idx]
                test_x = np.vstack([train_h5[self._EXAMPLE][client_id_test][self._IMGAE][()]])
                test_y = np.concatenate([train_h5[self._EXAMPLE][client_id_test][self._LABEL][()]])

        # preprocess
        if self.transform == "none":
            # static `TensorDataset`, the old behavior
            transform = _data_transforms_fed_cifar(self.n_class, train=True)
            train_x = transform(
                # channel last to channel first
                torch.div(torch.from_numpy(train_x).permute(0, 3, 1, 2), 255.0)
            )
            train_y = torch.from_numpy(train_y).long()
            train_ds = torchdata.TensorDataset(train_x, train_y)
        else:
            # use non-trivial dynamic transform
            train_ds = VisionDataset(
                # channel last to channel first
                images=torch.from_numpy(train_x).permute(0, 3, 1, 2).to(torch.uint8),
                targets=torch.from_numpy(train_y).long(),
                transform=self.transform,
            )

        if len(test_x) != 0:
            # test dataset is always a static `TensorDataset`
            # with only normalization transform
            # and without any augmentation transform
            transform = _data_transforms_fed_cifar(self.n_class, train=False)
            test_x = transform(
                # channel last to channel first
                torch.div(torch.from_numpy(test_x).permute(0, 3, 1, 2), 255.0)
            )
            test_y = torch.from_numpy(test_y).long()
            test_ds = torchdata.TensorDataset(test_x, test_y)

        # generate dataloader
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        if len(test_x) != 0:
            test_dl = torchdata.DataLoader(
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
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """A set of candidate models."""
        return {
            "cnn_cifar": mnn.CNNCifar(num_classes=self.n_class),
            "cnn_cifar_small": mnn.CNNCifar_Small(num_classes=self.n_class),
            "cnn_cifar_tiny": mnn.CNNCifar_Tiny(num_classes=self.n_class),
            "resnet10": mnn.ResNet10(num_classes=self.n_class),
        }

    @property
    def doi(self) -> str:
        """DOI(s) related to the dataset."""
        return [
            "10.48550/ARXIV.2007.13518",  # FedML
        ]

    @property
    def label_map(self) -> dict:
        """Label map for the dataset."""
        return {
            10: CIFAR10_LABEL_MAP,
            100: CIFAR100_FINE_LABEL_MAP,
        }[self.n_class]

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

        if client_idx >= len(self._client_ids_train):
            raise ValueError(f"client_idx should be less than {len(self._client_ids_train)}")
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
        plt.title(f"client_id: {client_id}, label: {label} ({self.label_map[int(label)]})")
        plt.show()

    def random_grid_view(self, nrow: int, ncol: int, save_path: Optional[Union[str, Path]] = None) -> None:
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

        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1, nrow * 1))
        selected = []
        for i in range(nrow):
            for j in range(ncol):
                while True:
                    client_idx = rng.integers(len(self._client_ids_train))
                    client_id = self._client_ids_train[client_idx]
                    tot_img = train_h5[self._EXAMPLE][client_id][self._IMGAE][()]
                    image_idx = rng.integers(len(tot_img))
                    if (client_idx, image_idx) not in selected:
                        selected.append((client_idx, image_idx))
                        break
                img = tot_img[image_idx]
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.tight_layout()
        plt.show()


@register_fed_dataset()
class FedCIFAR100(FedCIFAR):
    __name__ = "FedCIFAR100"

    def __init__(
        self,
        datadir: Optional[Union[str, Path]] = None,
        transform: Optional[Union[str, Callable]] = "none",
        seed: int = 0,
        **extra_config: Any,
    ) -> None:
        super().__init__(100, datadir, transform, seed, **extra_config)

    @property
    def url(self) -> str:
        """URL for downloading the dataset."""
        return "https://fedml.s3-us-west-1.amazonaws.com/fed_cifar100.tar.bz2"


def _data_transforms_fed_cifar(
    n_class: int,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    train: bool = True,
    crop_size: Sequence[int] = (24, 24),
) -> Callable:
    """Get data transforms for CIFAR10/100 dataset.

    Parameters
    ----------
    n_class : int
        Number of classes in the dataset.
        10 for CIFAR10, 100 for CIFAR100.
    mean : Sequence[float], optional
        Mean for normalization.
        If ``None``, use default mean.
    std : Sequence[float], optional
        Standard deviation for normalization.
        If ``None``, use default standard deviation.
    train : bool, default True
        Whether to get training transforms.
    crop_size : Sequence[int], default (24, 24)
        Crop size for random crop.

    Returns
    -------
    Callable
        Transforms to apply to the images.

    """
    assert n_class in [10, 100]
    if mean is None:
        mean = CIFAR10_MEAN if n_class == 10 else CIFAR100_MEAN
    if std is None:
        std = CIFAR10_STD if n_class == 10 else CIFAR100_STD
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


@register_fed_dataset()
class FedCIFAR100_LDA(FedCIFAR100):
    """Federated CIFAR-100 dataset partitioned by Latent Dirichlet Allocation (LDA).

    This class extends `FedCIFAR100` to create a non-IID data distribution
    among clients. Instead of using the default TFF partition, it loads the
    entire training dataset and repartitions it among a specified number of
    clients using a Dirichlet distribution, controlled by the `lda_alpha` parameter.

    A smaller `lda_alpha` value results in a more skewed, non-IID distribution,
    where each client may only have data from a few classes. A larger `lda_alpha`
    value leads to a more uniform, IID-like distribution.

    The test dataset remains global and is shared among all clients for evaluation.

    Parameters
    ----------
    num_clients : int
        The total number of clients to partition the data for.
    lda_alpha : float
        The concentration parameter for the Dirichlet distribution. Controls the
        degree of non-IID-ness.
    datadir : str or pathlib.Path, optional
        Path to the dataset directory. If ``None``, uses the default cache path.
    transform : str or callable, optional
        Transformation to apply to the images. Defaults to ``"none"``.
    seed : int, optional
        Random seed for data shuffling and partitioning. Defaults to 0.
    **extra_config : dict, optional
        Extra configurations for the dataset.

    """

    __name__ = "FedCIFAR100_LDA"

    def __init__(
        self,
        num_clients: int,
        lda_alpha: float,
        datadir: Optional[Union[str, Path]] = None,
        transform: Optional[Union[str, Callable]] = "none",
        seed: int = 0,
        **extra_config: Any,
    ) -> None:
        # Call the parent constructor to handle data downloading, default paths, etc.
        super().__init__(datadir=datadir, transform=transform, seed=seed, **extra_config)

        self.num_clients = num_clients
        self.lda_alpha = lda_alpha

        # Set the random seed for reproducible partitioning
        np.random.seed(self.seed)

        print("Partitioning data using LDA. This may take a moment...")
        # Step 1: Load all training labels from the HDF5 file to perform the partition.
        train_h5_path = self.datadir / self.DEFAULT_TRAIN_FILE
        with h5py.File(str(train_h5_path), "r") as train_h5:
            # The original data is partitioned into 500 clients. We merge them.
            all_train_labels = np.concatenate(
                [train_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in self._client_ids_train]
            )

        # Step 2: Use the LDA utility function to get the partition map.
        # The map is a dictionary {client_idx: [sample_indices]}
        self.partition_map = non_iid_partition_with_dirichlet_distribution(
            label_list=all_train_labels,
            client_num=self.num_clients,
            classes=self.n_class,
            alpha=self.lda_alpha,
            task="classification",
        )

        # Step 3 (Optional but recommended): Record the data statistics for analysis.
        self.data_statistics = record_data_stats(all_train_labels, self.partition_map)
        print("Data partitioning complete.")

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        """Get local dataloader for a client created by LDA partition.

        This method overrides the parent implementation. It loads the entire
        training dataset and then selects the subset of data for the specified
        `client_idx` based on the pre-computed LDA partition map.

        The test dataloader contains the complete, global test set, which is
        standard practice for evaluation in federated learning.

        Parameters
        ----------
        train_bs : int, optional
            Batch size for training dataloader. If ``None``, use default batch size.
        test_bs : int, optional
            Batch size for testing dataloader. If ``None``, use default batch size.
        client_idx : int, optional
            Index of the client (from 0 to `num_clients` - 1).
            If ``None``, returns a dataloader for the entire dataset for centralized training.

        Returns
        -------
        train_dl : torch.utils.data.DataLoader
            Training dataloader for the specified client.
        test_dl : torch.utils.data.DataLoader
            Global testing dataloader.

        """
        if client_idx is None:
            # For centralized training, fall back to the parent's behavior.
            return super().get_dataloader(train_bs, test_bs, None)

        if not (0 <= client_idx < self.num_clients):
            raise ValueError(f"`client_idx` must be between 0 and {self.num_clients - 1}, but got {client_idx}.")

        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")

        # --- Load entire dataset into memory ---
        # This is necessary because LDA partition gives indices over the whole dataset.
        # For CIFAR-100, this is feasible (50000 images, ~150MB).
        all_train_x = np.vstack([train_h5[self._EXAMPLE][cid][self._IMGAE][()] for cid in self._client_ids_train])
        all_train_y = np.concatenate([train_h5[self._EXAMPLE][cid][self._LABEL][()] for cid in self._client_ids_train])

        all_test_x = np.vstack([test_h5[self._EXAMPLE][cid][self._IMGAE][()] for cid in self._client_ids_test])
        all_test_y = np.concatenate([test_h5[self._EXAMPLE][cid][self._LABEL][()] for cid in self._client_ids_test])

        # --- Slice data for the specific client using the LDA partition map ---
        client_sample_indices = self.partition_map[client_idx]
        train_x = all_train_x[client_sample_indices]
        train_y = all_train_y[client_sample_indices]

        # --- Create Datasets and DataLoaders ---
        # The logic below is adapted from the parent class to ensure consistency.

        # Create training dataset
        if self.transform == "none":
            # Apply static normalization (old behavior).
            transform = _data_transforms_fed_cifar(self.n_class, train=True)
            train_x_tensor = transform(
                # Permute from HWC (channel last) to CHW (channel first) and normalize.
                torch.div(torch.from_numpy(train_x).permute(0, 3, 1, 2), 255.0)
            )
            train_y_tensor = torch.from_numpy(train_y).long()
            train_ds = torchdata.TensorDataset(train_x_tensor, train_y_tensor)
        else:
            # Use dynamic transforms (e.g., augmentations).
            train_ds = VisionDataset(
                images=torch.from_numpy(train_x).permute(0, 3, 1, 2).to(torch.uint8),
                targets=torch.from_numpy(train_y).long(),
                transform=self.transform,
            )

        # Create test dataset (always static with normalization only).
        test_transform = _data_transforms_fed_cifar(self.n_class, train=False)
        test_x_tensor = test_transform(
            torch.div(torch.from_numpy(all_test_x).permute(0, 3, 1, 2), 255.0)
        )
        test_y_tensor = torch.from_numpy(all_test_y).long()
        test_ds = torchdata.TensorDataset(test_x_tensor, test_y_tensor)

        # Create DataLoaders.
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_dl = torchdata.DataLoader(
            dataset=test_ds,
            batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=False,  # Test set is typically not shuffled.
            drop_last=False,
        )

        train_h5.close()
        test_h5.close()

        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        """Add LDA-specific parameters to the class representation."""
        return super().extra_repr_keys() + ["num_clients", "lda_alpha"]