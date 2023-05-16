"""
Modified from `generate_synthetic.py` in `FedProx`

ref. https://github.com/litian96/FedProx/blob/master/data/

"""

import itertools
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch  # noqa: F401
from scipy.io import loadmat, savemat

from ..utils.misc import CACHED_DATA_DIR


__all__ = [
    "generate_synthetic",
]


(CACHED_DATA_DIR / "synthetic").mkdir(parents=True, exist_ok=True)
_NAME_PATTERN = (
    "synthetic_{alpha}_{beta}_{iid}_{num_clients}_{num_classes}_{dimension}_{seed}.mat"
)


def generate_synthetic(
    alpha: float,
    beta: float,
    iid: bool,
    num_clients: int,
    num_classes: int = 10,
    dimension: int = 60,
    seed: int = 0,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    recompute: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Generate synthetic data using methods proposed in FedProx paper."""
    file = _get_path(alpha, beta, iid, num_clients, num_classes, dimension, seed)
    if recompute or not file.exists():
        data_dict = _generate_synthetic(
            alpha,
            beta,
            iid,
            num_clients,
            num_classes,
            dimension,
            seed,
        )
        savemat(str(file), data_dict)
    else:
        data_dict = loadmat(str(file))
    split_inds = data_dict["split"]
    samples_per_client = split_inds[..., 1] - split_inds[..., 0]
    if shuffle:
        shuffled_inds = [np.random.permutation(range(n)) for n in samples_per_client]
    else:
        shuffled_inds = [np.arange(n) for n in samples_per_client]
    clients = [
        {
            "train_X": data_dict["X"][spl_i[0] : spl_i[1]][shf_i][
                : int(train_ratio * n)
            ],
            "train_y": data_dict["y"].flatten()[spl_i[0] : spl_i[1]][shf_i][
                : int(train_ratio * n)
            ],
            "test_X": data_dict["X"][spl_i[0] : spl_i[1]][shf_i][
                int(train_ratio * n) :
            ],
            "test_y": data_dict["y"].flatten()[spl_i[0] : spl_i[1]][shf_i][
                int(train_ratio * n) :
            ],
        }
        for n, spl_i, shf_i in zip(samples_per_client, split_inds, shuffled_inds)
    ]
    return clients


def _generate_synthetic(
    alpha: float,
    beta: float,
    iid: bool,
    num_clients: int,
    num_classes: int = 10,
    dimension: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples_per_client = rng.lognormal(4, 2, (num_clients)).astype(int) + 50
    num_samples = np.sum(samples_per_client)
    X_split = list(itertools.repeat([], num_clients))
    y_split = list(itertools.repeat([], num_clients))

    mean_W = rng.normal(0, alpha, num_clients)
    mean_b = mean_W
    B = rng.normal(0, beta, num_clients)
    mean_x = np.zeros((num_clients, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(num_clients):
        if iid:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = rng.normal(B[i], 1, dimension)

    if iid:
        W_global = rng.normal(0, 1, (dimension, num_classes))
        b_global = rng.normal(0, 1, num_classes)

    for i in range(num_clients):
        if iid:
            W = W_global
            b = b_global
        else:
            W = rng.normal(mean_W[i], 1, (dimension, num_classes))
            b = rng.normal(mean_b[i], 1, num_classes)

        xx = rng.multivariate_normal(mean_x[i], cov_x, samples_per_client[i]).astype(
            np.float32
        )
        yy = np.zeros(samples_per_client[i], dtype=int)

        for j in range(samples_per_client[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(torch.softmax(torch.from_numpy(tmp), dim=0).numpy())

        X_split[i] = xx
        y_split[i] = yy

        # print(f"{i}-th client has {len(y_split[i])} exampls")

    split_inds = np.cumsum(samples_per_client)
    split_inds = np.array([np.append([0], split_inds[:-1]), split_inds]).T
    data_dict = {
        "X": np.concatenate(X_split, axis=0),
        "y": np.concatenate(y_split, axis=0),
        "split": split_inds,
    }

    return data_dict


def _get_path(
    alpha: float,
    beta: float,
    iid: bool,
    num_clients: int,
    num_classes: int = 10,
    dimension: int = 60,
    seed: int = 0,
) -> Path:
    return (
        CACHED_DATA_DIR
        / "synthetic"
        / _NAME_PATTERN.format(
            alpha=alpha,
            beta=beta,
            iid="iid" if iid else "noniid",
            num_clients=num_clients,
            num_classes=num_classes,
            dimension=dimension,
            seed=seed,
        )
    )
