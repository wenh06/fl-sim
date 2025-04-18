import json
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
from torch_ecg.utils import ReprMixin
from torchvision.datasets import CIFAR10, CIFAR100

from ..utils.const import CACHED_DATA_DIR, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from ..utils.misc import set_seed

__all__ = [
    "CIFAR_truncated",
    "CIFAR10_truncated",
    "CIFAR100_truncated",
    "load_cifar_data",
    "partition_cifar_data",
]

CIFAR_DATA_DIRS = {
    n_class: (CACHED_DATA_DIR / f"CIFAR{n_class}")
    for n_class in [
        10,
        100,
    ]
}
CIFAR_NONIID_CACHE_DIRS = {
    n_class: (CACHED_DATA_DIR / "non-iid-distribution" / f"CIFAR{n_class}")
    for n_class in [
        10,
        100,
    ]
}
for n_class in [
    10,
    100,
]:
    CIFAR_DATA_DIRS[n_class].mkdir(parents=True, exist_ok=True)
    CIFAR_NONIID_CACHE_DIRS[n_class].mkdir(parents=True, exist_ok=True)


class CIFAR_truncated(ReprMixin, torchdata.Dataset):
    """Truncated CIFAR dataset.

    This class is modified from `FedML <https://github.com/FedML-AI/FedML>`_.

    Parameters
    ----------
    n_class : {10, 100}
        Number of classes.
    root : str or pathlib.Path, optional
        Directory to store the data,
        defaults to pre-defined directory for CIFAR data.
    dataidxs : list of int, optional
        List of indices of the data to be used.
    train : bool, default True
        Whether to use training data.
    transform : callable, optional
        Transform to apply to the data.
    target_transform : callable, optional
        Transform to apply to the targets (labels).
    download : bool, default False
        Whether to download the data if not found locally.
    seed : int, default 0
        Random seed for reproducibility.

    """

    __name__ = "CIFAR10_truncated"

    def __init__(
        self,
        n_class: Literal[10, 100] = 10,
        root: Optional[Union[str, Path]] = None,
        dataidxs: Optional[List[int]] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        seed: int = 0,
    ) -> None:
        self.n_class = n_class
        self.root = Path(root or CIFAR_DATA_DIRS[n_class])
        assert self.n_class in [10, 100]
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.seed = seed
        set_seed(seed=self.seed)

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the truncated dataset via
        filtering the data with the given `dataidxs`."""
        DS = {10: CIFAR10, 100: CIFAR100}[self.n_class]
        cifar_dataobj = DS(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index: np.ndarray) -> None:
        """Truncate the channel of the image.

        Parameters
        ----------
        index : numpy.ndarray
            Indices of the samples to be truncated.

        Returns
        -------
        None

        """
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the sample given the index.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        img : torch.Tensor
            The image.
        target : torch.Tensor
            The label.

        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "n_class",
            "root",
        ]


class CIFAR10_truncated(CIFAR_truncated):
    """Truncated CIFAR10 dataset.

    Parameters
    ----------
    root : str or pathlib.Path, optional
        Directory to store the data,
        defaults to pre-defined directory for CIFAR data.
    dataidxs : list of int, optional
        List of indices of the data to be used.
    train : bool, default True
        Whether to use training data.
    transform : callable, optional
        Transform to apply to the data.
    target_transform : callable, optional
        Transform to apply to the targets (labels).
    download : bool, default False
        Whether to download the data if not found locally.

    """

    __name__ = "CIFAR10_truncated"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        dataidxs: Optional[List[int]] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(10, root, dataidxs, train, transform, target_transform, download)


class CIFAR100_truncated(CIFAR_truncated):
    """Truncated CIFAR100 dataset.

    Parameters
    ----------
    root : str or pathlib.Path, optional
        Directory to store the data,
        defaults to pre-defined directory for CIFAR data.
    dataidxs : list of int, optional
        List of indices of the data to be used.
    train : bool, default True
        Whether to use training data.
    transform : callable, optional
        Transform to apply to the data.
    target_transform : callable, optional
        Transform to apply to the targets (labels).
    download : bool, default False
        Whether to download the data if not found locally.

    """

    __name__ = "CIFAR100_truncated"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        dataidxs: Optional[List[int]] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(100, root, dataidxs, train, transform, target_transform, download)


class Cutout(object):
    """The cutout augmentation.

    Parameters
    ----------
    length : int
        The length of the cutout square.

    """

    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        img of shape ``[..., H, W]``
        """
        h, w = img.shape[-2:]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar(n_class: Literal[10, 100]) -> Tuple[Callable, Callable]:
    """Get data transforms for CIFAR data.

    Parameters
    ----------
    n_class : {10, 100}
        Number of classes.

    Returns
    -------
    train_transform : Callable
        Transform for training data.
    test_transform : Callable
        Transform for testing data.

    """
    if n_class == 10:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif n_class == 100:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"n_class should be in [10, 100], got {n_class}")
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std),
            Cutout(16),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, test_transform


def load_cifar_data(
    n_class: int, datadir: Optional[Union[str, Path]] = None, to_numpy: bool = False
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    """Load CIFAR data with preprocessing.

    Parameters
    ----------
    n_class : int
        Number of classes.
    datadir : str or pathlib.Path, optional
        Directory to store the data,
        defaults to pre-defined directory for CIFAR data.
    to_numpy : bool, default False
        Whether to convert the data to numpy array.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test)``

    """
    train_transform, test_transform = _data_transforms_cifar(n_class)

    cifar_train_ds = CIFAR_truncated(datadir, n_class, train=True, download=True, transform=train_transform)
    cifar_test_ds = CIFAR_truncated(datadir, n_class, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar_train_ds.data, cifar_train_ds.target
    X_test, y_test = cifar_test_ds.data, cifar_test_ds.target

    if to_numpy:
        (
            X_train.cpu().numpy(),
            y_train.cpu().numpy(),
            X_test.cpu().numpy(),
            y_test.cpu().numpy(),
        )
    else:
        return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train: torch.Tensor, net_dataidx_map: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    """Record the number of training samples for each class on each client.

    Parameters
    ----------
    y_train : torch.Tensor
        Training labels.
    net_dataidx_map : Dict[int, List[int]]
        Mapping from client index to the list of training sample indices.

    Returns
    -------
    Dict[int, Dict[int, int]]
        Number of training samples for each class on each client.

    """
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx].cpu().numpy(), return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def partition_cifar_data(
    dataset: CIFAR_truncated,
    partition: str,
    n_net: int,
    alpha: float,
    to_numpy: bool = False,
    datadir: Optional[Union[str, Path]] = None,
) -> tuple:
    """Partition CIFAR data.

    Parameters
    ----------
    dataset : type
        CIFAR (10 or 100) dataset class.
    partition : str
        Partition method.
    n_net : int
        Number of clients.
    alpha : float
        Parameter for Dirichlet distribution.
    to_numpy : bool, default False
        Whether to convert the data to numpy array.
    datadir : str or pathlibPath, optional
        Directory to store the data,
        defaults to pre-defined directory for CIFAR data.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)``
        where ``net_dataidx_map`` is a dict mapping from client index to the list of
        training sample indices, and ``traindata_cls_counts`` is a dict mapping from
        client index to the number of training samples for each class.

    """
    n_class = dataset.n_class
    X_train, y_train, X_test, y_test = load_cifar_data(n_class, datadir, to_numpy)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_net)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_net)}
    elif partition == "hetero":
        min_size = 0
        K = 100
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_net)]
            # for each class in the dataset
            for k in range(K):
                if to_numpy:
                    idx_k = np.where(y_train == k)[0]
                else:
                    idx_k = np.where(y_train.cpu().numpy() == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_net))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / n_net) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_net):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = CIFAR_NONIID_CACHE_DIRS[n_class] / "net_dataidx_map.json"
        with open(dataidx_map_file_path, "r") as f:
            net_dataidx_map = json.load(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = CIFAR_NONIID_CACHE_DIRS[n_class] / "distribution.json"
        with open(distribution_file_path, "r") as f:
            traindata_cls_counts = json.load(distribution_file_path)
    else:
        distribution_file_path = CIFAR_NONIID_CACHE_DIRS[n_class] / "distribution.json"
        with open(distribution_file_path, "w") as f:
            traindata_cls_counts = json.dump(traindata_cls_counts, distribution_file_path, ensure_ascii=False)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts
